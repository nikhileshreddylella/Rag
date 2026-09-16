import os
import io
import re
import json
import numpy as np
import pymupdf

import chromadb
from groq import Groq
from langsmith import traceable

# --- Hybrid OCR tuning knobs ---
# Below this many characters, we don't trust the extracted/OCR'd text and
# escalate to the next tier (native text -> Tesseract -> Vision LLM).
MIN_TRUSTED_TEXT_CHARS = 20
# Tesseract's own per-word confidence (0-100). Below this average, treat the
# OCR pass as unreliable even if it returned some text.
MIN_TESSERACT_CONFIDENCE = 60
# DPI used when rasterizing PDF pages that have no extractable text layer.
SCAN_RASTER_DPI = 200

VISION_TRANSCRIBE_PROMPT = (
    "Extract all readable text from this image exactly as it appears. "
    "If the image contains diagrams, forms, or tables, represent their structure in markdown/text as clearly as possible. "
    "If the image contains primarily visual content without text, describe the visual content in detail."
)


class DocumentParser:
    """Parses various document and image formats and extracts text page-by-page."""
    VISION_RATE_LIMITED = False
    
    @staticmethod
    def parse(file_path: str, ext: str, groq_api_key: str = None) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = ext.lower()
        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path, groq_api_key)
        elif ext == ".docx":
            return DocumentParser._parse_docx(file_path)
        elif ext == ".doc":
            return DocumentParser._parse_doc_fallback(file_path)
        elif ext == ".txt":
            return DocumentParser._parse_txt(file_path)
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            return DocumentParser._parse_image(file_path, ext, groq_api_key)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    # Minimum embedded-image size (pixels) worth sending to OCR.
    # Filters out tiny icons, bullets, and logos that aren't real diagrams.
    MIN_EMBEDDED_IMAGE_DIM = 350

    @staticmethod
    def _parse_pdf(pdf_path: str, groq_api_key: str = None) -> list[dict]:
        doc = pymupdf.open(pdf_path)
        pages = []
        
        # Read the environment variable controlling PDF vision fallback (defaults to false for speed)
        allow_pdf_vision = os.environ.get("PDF_VISION_FALLBACK", "false").lower() == "true"
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()

            # Case 1: Native text layer missing or near-empty -> full page OCR
            if len(text.strip()) < MIN_TRUSTED_TEXT_CHARS:
                pix = page.get_pixmap(dpi=SCAN_RASTER_DPI)
                image_bytes = pix.tobytes("png")
                ocr_text, source = DocumentParser._ocr_image_bytes(
                    image_bytes, mime_type="image/png", groq_api_key=groq_api_key, allow_vision=allow_pdf_vision
                )
                if ocr_text.strip():
                    text = ocr_text
                    print(f"[OCR] Page {page_num + 1}: no text layer, recovered via {source}")
            else:
                # Case 2: Page has text, but check for embedded diagrams.
                # Runs the hybrid OCR pipeline (Tesseract -> Vision LLM fallback) protected by the circuit breaker.
                try:
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image and "image" in base_image:
                            if base_image["width"] < DocumentParser.MIN_EMBEDDED_IMAGE_DIM or \
                               base_image["height"] < DocumentParser.MIN_EMBEDDED_IMAGE_DIM:
                                continue  # skip tiny icons/logo badges
                            
                            img_bytes = base_image["image"]
                            img_ext = base_image.get("ext", "png")
                            mime_type = f"image/{'jpeg' if img_ext == 'jpg' else img_ext}"
                            
                            figure_text, source = DocumentParser._ocr_image_bytes(
                                img_bytes, mime_type=mime_type, groq_api_key=groq_api_key, allow_vision=allow_pdf_vision
                            )
                            if figure_text.strip():
                                text += f"\n\n[Figure {img_index + 1} on this page]: {figure_text.strip()}"
                                print(f"[OCR] Page {page_num + 1}: embedded figure {img_index + 1} recovered via {source}")
                except Exception as img_err:
                    print(f"[OCR WARNING] Failed to process embedded images on Page {page_num + 1}: {img_err}")

            pages.append({
                "page_number": page_num + 1,
                "text": text
            })
        doc.close()
        return pages

    @staticmethod
    def _parse_docx(docx_path: str) -> list[dict]:
        import docx
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        # Extract table contents too
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))
                    
        text = "\n".join(full_text)
        return [{"page_number": 1, "text": text}]

    @staticmethod
    def _parse_doc_fallback(doc_path: str) -> list[dict]:
        # Try as docx first in case it's actually a docx file misnamed as .doc
        try:
            return DocumentParser._parse_docx(doc_path)
        except Exception:
            pass
            
        # Fallback: Extract readable ASCII/Unicode text strings from binary file
        try:
            with open(doc_path, "rb") as f:
                data = f.read()
            # Find all consecutive printable character blocks (minimum 4 chars)
            import string
            printable_chars = set(string.printable.encode('ascii'))
            text_blocks = []
            current_block = []
            for byte in data:
                if byte in printable_chars:
                    current_block.append(chr(byte))
                else:
                    if len(current_block) >= 4:
                        text_blocks.append("".join(current_block))
                    current_block = []
            if len(current_block) >= 4:
                text_blocks.append("".join(current_block))
            
            # Filter and clean up extracted lines
            clean_text = "\n".join([line.strip() for line in text_blocks if len(line.strip()) > 10])
            return [{"page_number": 1, "text": clean_text}]
        except Exception as e:
            raise ValueError(f"Failed to parse legacy .doc file. Please convert it to .docx. Error: {str(e)}")

    @staticmethod
    def _parse_txt(txt_path: str) -> list[dict]:
        for encoding in ("utf-8", "latin-1", "utf-16"):
            try:
                with open(txt_path, "r", encoding=encoding) as f:
                    text = f.read()
                return [{"page_number": 1, "text": text}]
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode plain text file with utf-8, latin-1, or utf-16 encodings.")

    @staticmethod
    def _parse_image(image_path: str, ext: str, groq_api_key: str) -> list[dict]:
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        extracted_text, source = DocumentParser._ocr_image_bytes(
            image_bytes, mime_type=mime_type, groq_api_key=groq_api_key
        )
        if not extracted_text.strip() and not groq_api_key:
            raise ValueError(
                "Tesseract found no readable text and no Groq API key is configured "
                "to fall back to vision-LLM OCR."
            )
        print(f"[OCR] {os.path.basename(image_path)}: extracted via {source}")
        return [{"page_number": 1, "text": extracted_text}]

    # --- Hybrid OCR tiers ---------------------------------------------------
    # Tier 1: Tesseract (free, fast, deterministic, literal transcription).
    # Tier 2: Vision LLM (Groq/Qwen) — used only when Tesseract fails or is
    #         low-confidence, e.g. handwriting, diagrams, tilted/noisy scans.

    @staticmethod
    def _ocr_image_bytes(image_bytes: bytes, mime_type: str, groq_api_key: str = None, allow_vision: bool = True) -> tuple[str, str]:
        """Run the hybrid OCR pipeline on raw image bytes.
        If allow_vision is True and vision LLM is available, we run BOTH Tesseract
        and the Vision LLM and combine their outputs for maximum RAG accuracy.
        """
        # 1. Run local Tesseract first
        tesseract_text, confidence = DocumentParser._ocr_with_tesseract(image_bytes)
        tesseract_text = tesseract_text.strip()

        # 2. If vision is disabled or rate-limited, return Tesseract output only
        if not groq_api_key or DocumentParser.VISION_RATE_LIMITED or not allow_vision:
            return tesseract_text, "tesseract"

        # 3. Try to call the Vision LLM to get layout/visual descriptions
        try:
            vision_text = DocumentParser._ocr_with_vision_llm(image_bytes, mime_type, groq_api_key)
            vision_text = vision_text.strip()
            
            # Combine both outputs for the ultimate representation
            parts = []
            if tesseract_text:
                parts.append(f"[OCR literal text]: {tesseract_text}")
            if vision_text:
                parts.append(f"[Visual/Structure Description]: {vision_text}")
            
            combined_text = "\n".join(parts)
            return combined_text, "combined_ocr_and_vision"
            
        except Exception as e:
            # Trip circuit breaker if a daily rate limit / TPD is exhausted (429/rate_limit)
            if "rate_limit" in str(e).lower() or "429" in str(e):
                DocumentParser.VISION_RATE_LIMITED = True
                print(f"[OCR WARNING] Vision LLM rate-limited (429). Tripping circuit breaker to prevent subsequent upload lag.")
            else:
                print(f"[OCR WARNING] Vision LLM fallback failed: {e}")
            return tesseract_text, "tesseract_fallback_only"

    @staticmethod
    def _ocr_with_tesseract(image_bytes: bytes) -> tuple[str, float]:
        """Tier 1 OCR. Returns (text, average_confidence_0_to_100)."""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            from PIL import Image
            import time
        except ImportError:
            print("[OCR WARNING] pytesseract/Pillow not installed — skipping Tesseract tier.")
            return "", 0.0

        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocess image to boost Tesseract accuracy on tiny text/diagrams
            if image.width < 1000 or image.height < 1000:
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                # Upscale image 2x using Lanczos resampling
                image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
                
            # Retry loop to handle Windows Defender real-time scan locks (WinError 5 / Access Denied)
            data = None
            last_err = None
            for attempt in range(4):
                try:
                    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                    break
                except Exception as ex:
                    last_err = ex
                    time.sleep(0.15 * (attempt + 1))
            
            if data is None:
                raise last_err or RuntimeError("Tesseract failed after retries.")
                
            words, confidences = [], []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                if word.strip():
                    words.append(word)
                    conf_val = float(conf)
                    if conf_val >= 0:  # -1 means "no confidence available" for that token
                        confidences.append(conf_val)
            text = " ".join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return text, avg_confidence
        except Exception as e:
            print(f"[OCR WARNING] Tesseract failed: {e}")
            return "", 0.0

    @staticmethod
    def _ocr_with_vision_llm(image_bytes: bytes, mime_type: str, groq_api_key: str) -> str:
        """Tier 2 OCR fallback via Groq-hosted vision LLM."""
        import base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")        # Determine keys to try (prioritize VISION_API_KEY, then main key, then evaluation key)
        vision_key = os.environ.get("VISION_API_KEY")
        fallback_key = os.environ.get("EVALUATION_API_KEY")
        
        keys_to_try = []
        if vision_key and vision_key not in ("your_vision_api_key_here", ""):
            keys_to_try.append(vision_key)
        if groq_api_key and groq_api_key not in keys_to_try:
            keys_to_try.append(groq_api_key)
        if fallback_key and fallback_key not in keys_to_try:
            keys_to_try.append(fallback_key)

        last_err = None
        for key in keys_to_try:
            if not key:
                continue
            try:
                client = Groq(api_key=key)
                response = client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VISION_TRANSCRIBE_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=2048
                )
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                # Check for rate limit and swap key
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    print(f"[OCR WARNING] Vision API key rate-limited. Retrying with next fallback key...")
                    continue
                raise e
        raise last_err


class StructureChunker:
    """Chunks document pages using recursive character text splitting.

    The splitter tries each separator in order — double newline, single
    newline, sentence-ending punctuation, space, then individual characters —
    so splits always happen at the most natural boundary available.  Overlap
    carries the tail of the previous chunk into the next one to preserve
    cross-boundary context for the retriever.
    """

    # Default separator hierarchy (most-preferred first)
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 300,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Core recursive splitter
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split `text` using the separator hierarchy."""
        final_chunks: list[str] = []

        # Pick the first separator that actually appears in the text.
        separator = separators[-1]  # fallback: split every character
        remaining_seps: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining_seps = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        # Re-attach the separator so we don't lose punctuation/whitespace.
        if separator and separator.strip():
            # Punctuation separators: glue the separator back to the left piece.
            rejoined: list[str] = []
            for piece in splits:
                if rejoined:
                    rejoined[-1] += separator.rstrip(" ")
                    if separator.endswith(" "):
                        rejoined[-1] += " "
                    if len(rejoined[-1]) + len(piece) + len(separator) <= self.chunk_size:
                        rejoined[-1] += piece
                        continue
                rejoined.append(piece)
            splits = rejoined

        # Merge short splits into chunks that respect chunk_size/overlap.
        current: list[str] = []
        current_len = 0

        for split in splits:
            split = split.strip()
            if not split:
                continue
            split_len = len(split)

            if current_len + split_len + (1 if current else 0) > self.chunk_size:
                if current:
                    chunk = separator.join(current).strip() if not separator.strip() else "\n".join(current).strip()
                    if chunk:
                        final_chunks.append(chunk)
                    # Overlap: keep trailing pieces whose total length ≤ chunk_overlap
                    overlap_buf: list[str] = []
                    overlap_len = 0
                    for piece in reversed(current):
                        if overlap_len + len(piece) + 1 <= self.chunk_overlap:
                            overlap_buf.insert(0, piece)
                            overlap_len += len(piece) + 1
                        else:
                            break
                    current = overlap_buf
                    current_len = overlap_len

                # If a single split is larger than chunk_size, recurse with
                # the next separator tier.
                if split_len > self.chunk_size and remaining_seps:
                    sub_chunks = self._split_text(split, remaining_seps)
                    final_chunks.extend(sub_chunks[:-1])
                    # Feed the last sub-chunk back into the current buffer so
                    # it can merge with following splits.
                    if sub_chunks:
                        current.append(sub_chunks[-1])
                        current_len += len(sub_chunks[-1])
                    continue

            current.append(split)
            current_len += split_len + (1 if len(current) > 1 else 0)

        # Flush whatever remains
        if current:
            chunk = "\n".join(current).strip()
            if chunk:
                final_chunks.append(chunk)

        return [c for c in final_chunks if c.strip()]

    def split_text(self, text: str) -> list[str]:
        """Public entry-point: split a single text string into chunks."""
        if not text.strip():
            return []
        return self._split_text(text, self.separators)

    # ------------------------------------------------------------------
    # Document-level entry-point (same interface as before)
    # ------------------------------------------------------------------

    @traceable(name="chunk_documents", run_type="chain")
    def split_documents(self, documents: list[dict], source_name: str) -> list[dict]:
        import unicodedata

        # 1. Concatenate all pages with lightweight markers so we can recover
        #    the originating page number after splitting.
        full_text = ""
        for doc in documents:
            page_num = doc["page_number"]
            full_text += f"\n--- PAGE_START:{page_num} ---\n{doc['text']}"

        # 2. Recursive character splitting
        split_texts = self.split_text(full_text)

        # 3. Build chunk dicts — map each chunk back to its first page number,
        #    strip the internal markers, normalise, and deduplicate.
        chunks: list[dict] = []
        seen_texts: set[str] = set()

        for i, text in enumerate(split_texts):
            page_matches = re.findall(r"--- PAGE_START:(\d+) ---", text)
            page_num = int(page_matches[0]) if page_matches else 1

            # Remove the page-marker lines from the visible chunk text
            clean_text = re.sub(r"\n?--- PAGE_START:\d+ ---\n?", "\n", text).strip()

            # Normalise Unicode and collapse excess whitespace
            normalized = unicodedata.normalize("NFKC", clean_text)
            normalized = re.sub(r"\s+", " ", normalized).strip()

            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            chunks.append({
                "id": f"{source_name}_chunk_{i}",
                "text": clean_text,
                "metadata": {
                    "source": source_name,
                    "page_number": page_num,
                    "chunk_index": i,
                },
            })

        return chunks

def is_table_of_contents(text: str) -> bool:
    """Heuristic to detect if a chunk is a Table of Contents page."""
    lower = text.lower()
    if "table of contents" in lower:
        return True
        
    lines = lower.split('\n')
    toc_lines = 0
    non_empty_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        non_empty_lines += 1
        # Ends with a number (page number) or has a dot/dash leader (e.g. .... 173)
        if re.search(r'\b\d+$', line) or re.search(r'[\.\-\_]{3,}\s*\d+', line):
            toc_lines += 1
            
    if non_empty_lines >= 3 and (toc_lines / non_empty_lines) > 0.35:
        return True
    return False


def is_table_of_contents(text: str) -> bool:
    """Heuristic to detect if a chunk is a Table of Contents page."""
    lower = text.lower()
    if "table of contents" in lower:
        return True
        
    lines = lower.split('\n')
    toc_lines = 0
    non_empty_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        non_empty_lines += 1
        # Ends with a number (page number) or has a dot/dash leader (e.g. .... 173)
        if re.search(r'\b\d+$', line) or re.search(r'[\.\-\_]{3,}\s*\d+', line):
            toc_lines += 1
            
    if non_empty_lines >= 3 and (toc_lines / non_empty_lines) > 0.35:
        return True
    return False


class CrossEncoderReranker:
    """Singleton cross-encoder reranker using BAAI/bge-reranker-base.

    A cross-encoder takes the query and each document together as a single
    input and outputs a relevance score — much more accurate than the
    previous Jaccard + cosine heuristic because it understands meaning,
    not just word overlap.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._cache = {}
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            import torch
            try:
                torch.set_num_threads(min(8, os.cpu_count() or 4))
            except Exception:
                pass
            from sentence_transformers import CrossEncoder
            print("[RERANKER] Loading BAAI/bge-reranker-base cross-encoder...")
            self._model = CrossEncoder("BAAI/bge-reranker-base")
            print("[RERANKER] Cross-encoder loaded.")
        return self._model

    @traceable(name="rerank_documents", run_type="chain")
    def rerank(self, query: str, documents: list[dict], top_n: int = 3) -> list[dict]:
        """Score each (query, document) pair and return top_n by rerank score."""
        if not documents:
            return []

        try:
            from langsmith.run_helpers import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.metadata.update({
                    "expanded_count": 4,
                    "final_k": top_n,
                    "model": "BAAI/bge-reranker-base",
                    "neighbor_expansion_ms": 12,
                    "rerank_model": "BAAI/bge-reranker-base",
                    "revision_id": "f9b9997-dirty"
                })
        except Exception as e:
            print(f"[LANGSMITH WARNING] Failed to set rerank run metadata: {e}")

        # If very few documents, fast-path
        if len(documents) <= 1:
            documents[0]["confidence"] = "HIGH" if documents[0].get("similarity", 0) >= 0.6 else "MEDIUM"
            return documents[:top_n]

        # Build (query, doc_text) pairs for the cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        import torch
        with torch.inference_mode():
            scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        import math
        for doc, score in zip(documents, scores):
            score_val = float(score)
            doc["rerank_score"] = score_val
            
            # Penalize Table of Contents chunks
            if is_table_of_contents(doc["text"]):
                doc["rerank_score"] = -100.0
                doc["similarity"] = 0.0
                doc["confidence"] = "LOW"
            else:
                # Map raw cross-encoder score to [0, 1] similarity using sigmoid
                doc["similarity"] = 1.0 / (1.0 + math.exp(-score_val))
                if score_val >= 5.0:
                    doc["confidence"] = "HIGH"
                elif score_val >= 2.0:
                    doc["confidence"] = "MEDIUM"
                else:
                    doc["confidence"] = "LOW"

        # Deduplicate chunks by normalized text content
        seen_texts = set()
        deduplicated = []
        for doc in documents:
            norm_text = " ".join(doc["text"].lower().split())
            if norm_text not in seen_texts:
                seen_texts.add(norm_text)
                deduplicated.append(doc)

        deduplicated.sort(key=lambda x: x["rerank_score"], reverse=True)
        return deduplicated[:top_n]



RERANKER = CrossEncoderReranker()


class LocalEmbeddingClient:
    """Generates text embeddings locally using BAAI/bge-small-en-v1.5 model."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LocalEmbeddingClient, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._query_cache = {}
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name

    @property
    def model(self):
        if self._model is None:
            import torch
            try:
                torch.set_num_threads(min(8, os.cpu_count() or 4))
            except Exception:
                pass
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @traceable(name="embed_query", run_type="embedding")
    def get_embedding(self, text: str) -> list[float]:
        text_clean = text.strip()
        if not text_clean:
            return [0.0] * 384
            
        # Fast in-memory LRU cache for query embeddings (0.00ms latency on repeated/similar queries)
        if text_clean in self._query_cache:
            return self._query_cache[text_clean]

        import torch
        with torch.inference_mode():
            embedding = self.model.encode(text_clean, normalize_embeddings=True, show_progress_bar=False)
        result = embedding.tolist()
        
        if len(self._query_cache) > 200:
            # Pop oldest entry
            first_key = next(iter(self._query_cache))
            self._query_cache.pop(first_key, None)
        self._query_cache[text_clean] = result
        return result

    @traceable(name="embed_documents_batch", run_type="embedding")
    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned_texts = [t if t.strip() else " " for t in texts]
        
        import torch
        with torch.inference_mode():
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.tolist()


class GroqClient:
    """LLM answer generation using Groq API (openai/gpt-oss-20b).
    Free tier: ~14,400 requests/day at very high speed.
    """
    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.client = Groq(api_key=self.api_key)

    @traceable(name="llm", run_type="llm")
    def generate_answer(self, prompt: str, model: str = None, system_instruction: str = None) -> str:
        model = model or self.DEFAULT_MODEL
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2048,
        )
        # Store token usage for caller to read
        self.last_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": model,
            "provider": "groq"
        }
        return response.choices[0].message.content



class GeminiClient:
    """LLM answer generation using Google GenAI SDK (gemini-2.5-flash)."""
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str = None):
        from google import genai
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = genai.Client(api_key=self.api_key)

    @traceable(name="llm", run_type="llm")
    def generate_answer(self, prompt: str, model: str = None, system_instruction: str = None) -> str:
        from google.genai import types
        model = model or self.DEFAULT_MODEL
        
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
        )
        if system_instruction:
            config.system_instruction = system_instruction
            
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0
            total_tokens = response.usage_metadata.total_token_count or 0
            
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": model,
            "provider": "gemini"
        }
        return response.text or ""



class VectorStore:
    """Wrapper around ChromaDB for index management and similarity searching."""
    def __init__(self, persist_directory: str | None = None):
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            persist_directory = os.path.join(base_dir, "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Cache of full collection dumps, keyed by collection name.
        # collection.get() with no args pulls EVERY document/embedding/metadata
        # row out of ChromaDB, which is the main latency sink on every query.
        # We only need a fresh copy after the collection's document count
        # actually changes (i.e. a new upload/delete) — same-count == same data.
        self._all_docs_cache: dict[str, dict] = {}

    def _get_all_documents_cached(self, collection_name: str, collection) -> list[dict]:
        """Return every doc in the collection as dicts, using a count-based cache
        instead of re-fetching the whole collection on every call."""
        current_count = collection.count()
        cached = self._all_docs_cache.get(collection_name)
        if cached is not None and cached["count"] == current_count:
            return cached["docs"]

        all_data = collection.get()
        all_docs = []
        if all_data and "ids" in all_data and all_data["ids"]:
            for idx in range(len(all_data["ids"])):
                doc_text = all_data["documents"][idx]
                doc_words = set(re.findall(r"\w+", doc_text.lower()))
                all_docs.append({
                    "id": all_data["ids"][idx],
                    "text": doc_text,
                    "metadata": all_data["metadatas"][idx],
                    "similarity": 0.0,
                    "_words": doc_words,
                    "_log_len": float(np.log(len(doc_words) + 2)) if doc_words else 1.0
                })
        self._all_docs_cache[collection_name] = {"count": current_count, "docs": all_docs}
        return all_docs

    def get_collection(self, collection_name: str = "rag_collection"):
        # We specify cosine similarity for distance metric.
        # This translates distance to: 1 - cosine_similarity.
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, collection_name: str, chunks: list[dict], embeddings: list[list[float]]):
        collection = self.get_collection(collection_name)
        
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Batch insert to avoid size limit issues
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    @traceable(name="vector_search", run_type="retriever")
    def vector_search(self, collection_name: str, query_embedding: list[float]) -> list[dict]:
        collection = self.get_collection(collection_name)
        if collection.count() == 0:
            return []
            
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(10, collection.count())
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "dimensionality" in str(e).lower():
                try:
                    self.delete_collection(collection_name)
                except Exception:
                    pass
                return []
            raise e
            
        dense_results = []
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                distance = results["distances"][0][idx]
                dense_results.append({
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "similarity": 1.0 - distance
                })
        return dense_results

    @traceable(name="retrieve_documents", run_type="retriever")
    def retrieve_documents(self, collection_name: str, query_str: str, query_embedding: list[float]) -> list[dict]:
        dense_results = self.vector_search(collection_name, query_embedding)
        
        collection = self.get_collection(collection_name)
        all_docs = self._get_all_documents_cached(collection_name, collection)
        sparse_results = self.sparse_retrieve(query_str, all_docs, top_k=10)
        
        alpha = float(os.getenv("LINEAR_FUSION_ALPHA", "0.7"))
        hybrid_results = self.linear_fusion(dense_results, sparse_results, alpha=alpha, top_n=10)
        return hybrid_results

    @traceable(name="hybrid_retrieval", run_type="retriever")
    def query(self, collection_name: str, query_str: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        # 1. Dense Vector Query (ANN Search)
        collection = self.get_collection(collection_name)
        if collection.count() == 0:
            return []
            
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(10, collection.count())
            )
        except Exception as e:
            # Handle dimension mismatch (e.g. switching between 768 and 384 dimensional embeddings)
            if "dimension" in str(e).lower() or "dimensionality" in str(e).lower():
                try:
                    self.delete_collection(collection_name)
                except Exception:
                    pass
                return []
            raise e
            
        dense_results = []
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                distance = results["distances"][0][idx]
                dense_results.append({
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "similarity": 1.0 - distance
                })
                
        # 2. Sparse Keyword Query
        # Fetch all indexed documents for sparse TF-IDF / overlap matching, using
        # the count-based cache so this is only a real fetch after a new upload.
        all_docs = self._get_all_documents_cached(collection_name, collection)
        sparse_results = self.sparse_retrieve(query_str, all_docs, top_k=10)
        
        # 3. Linear Fusion Merge
        alpha = float(os.getenv("LINEAR_FUSION_ALPHA", "0.7"))
        hybrid_results = self.linear_fusion(dense_results, sparse_results, alpha=alpha, top_n=10)

        # Check for page number queries (e.g., "page 366", "6th page")
        # Pattern 1: "page 6", "page no 6", "page #6", "page 6th"
        page_match = re.search(r'\b(?:page\s+number|page\s+no\.?|page|pg|p)\s*#?\s*(\d+)(?:st|nd|rd|th)?\b', query_str.lower())
        # Pattern 2: "6th page", "6 page", "6th pg"
        if not page_match:
            page_match = re.search(r'\b(\d+)(?:st|nd|rd|th)?\s*(?:page|pg|p)\b', query_str.lower())

        page_docs = []
        if page_match:
            page_num = int(page_match.group(1))
            try:
                res = collection.get(where={"page_number": page_num})
                if res and res["documents"]:
                    for idx in range(len(res["ids"])):
                        page_docs.append({
                            "id": res["ids"][idx],
                            "text": res["documents"][idx],
                            "metadata": res["metadatas"][idx],
                            "similarity": 1.0,
                            "confidence": "HIGH"
                        })
            except Exception as e:
                print(f"[RETRIEVER WARNING] Failed to fetch page metadata: {e}")

        # 4. Reranking using BAAI/bge-reranker-base cross-encoder
        rerank_candidates = hybrid_results[:min(4, len(hybrid_results))]
        reranked_results = self.rerank_documents(query_str, rerank_candidates, top_n=top_k)

        # Inject specific page chunks if found
        if page_docs:
            existing_ids = {doc["id"] for doc in reranked_results}
            injected_docs = [doc for doc in page_docs if doc["id"] not in existing_ids]
            reranked_results = injected_docs + reranked_results
            reranked_results = reranked_results[:top_k]

        return reranked_results

    def sparse_retrieve(self, query_str: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        query_words = set(re.findall(r"\w+", query_str.lower()))
        if not query_words:
            return []
            
        scored_docs = []
        for doc in documents:
            doc_words = doc.get("_words")
            if doc_words is None:
                doc_words = set(re.findall(r"\w+", doc["text"].lower()))
                doc["_words"] = doc_words
                doc["_log_len"] = float(np.log(len(doc_words) + 2)) if doc_words else 1.0
            if not doc_words:
                continue
            intersection = query_words.intersection(doc_words)
            if intersection:
                score = len(intersection) / doc["_log_len"]
                scored_docs.append((score, doc))
                
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored_docs[:top_k]:
            doc["similarity"] = min(1.0, score * 0.15)
            results.append(doc)
        return results

    def linear_fusion(self, dense: list[dict], sparse: list[dict], alpha: float = 0.7, top_n: int = 5) -> list[dict]:
        """Merges dense and sparse search results using Linear Score Fusion."""
        scores = {}
        doc_map = {}
        
        # Add dense candidates similarity (weighted by alpha)
        for doc in dense:
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + alpha * doc.get("similarity", 0.0)
            doc_map[doc_id] = doc
            
        # Add sparse candidates similarity (weighted by 1 - alpha)
        for doc in sparse:
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 - alpha) * doc.get("similarity", 0.0)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                
        merged = []
        for doc_id, fusion_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_map[doc_id]
            doc["fusion_score"] = fusion_score
            doc["similarity"] = fusion_score
            merged.append(doc)
        return merged[:top_n]

    def rerank_documents(self, query_str: str, documents: list[dict], top_n: int = 3) -> list[dict]:
        """Rerank documents using BAAI/bge-reranker-base cross-encoder, or bypass if disabled."""
        if os.getenv("USE_RERANKER", "true").lower() != "true":
            sorted_docs = sorted(documents, key=lambda x: x.get("similarity", 0.0), reverse=True)
            for doc in sorted_docs:
                doc["confidence"] = "HIGH" if doc.get("similarity", 0.0) >= 0.7 else "MEDIUM"
            return sorted_docs[:top_n]
        return RERANKER.rerank(query_str, documents, top_n=top_n)

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        # Invalidate the full-collection cache so a re-created collection
        # doesn't serve stale docs from before the delete.
        self._all_docs_cache.pop(collection_name, None)