path = '../frontend/script.js'
content = open(path, encoding='utf-8').read()

target1 = """// Submit user query to backend pipeline
function handleQuerySubmit(queryText, isInitial) {"""

replacement1 = """function setInputState(disabled) {
    const landingInput = document.getElementById("landing-search-input");
    const landingBtn = document.getElementById("btn-submit-landing");
    const chatInput = document.getElementById("chat-input");
    const chatBtn = document.getElementById("btn-submit-chat");
    
    if (landingInput) landingInput.disabled = disabled;
    if (landingBtn) landingBtn.disabled = disabled;
    if (chatInput) chatInput.disabled = disabled;
    if (chatBtn) chatBtn.disabled = disabled;
    
    if (chatInput) {
        if (disabled) {
            chatInput.placeholder = "Please wait until the answer is generated...";
        } else {
            chatInput.placeholder = "Write a message...";
        }
    }
}

// Submit user query to backend pipeline
function handleQuerySubmit(queryText, isInitial) {
    if (!queryText.trim()) return;

    // Prevent double submission if already processing
    const chatInput = document.getElementById("chat-input");
    if (chatInput && chatInput.disabled) return;"""

target2 = """    // Clear inputs
    document.getElementById("landing-search-input").value = "";
    document.getElementById("chat-input").value = "";"""

replacement2 = """    // Clear and disable inputs
    document.getElementById("landing-search-input").value = "";
    document.getElementById("chat-input").value = "";
    setInputState(true);"""

target3a = """            // Save to server
            saveSessionsToServer();

            // Re-render + refresh token counter
            renderActiveSessionMessages();
            renderRecentSessions();
            fetchTokenUsage();
        })"""

replacement3a = """            // Save to server
            saveSessionsToServer();

            // Re-render + refresh token counter
            renderActiveSessionMessages();
            renderRecentSessions();
            fetchTokenUsage();
            setInputState(false);
        })"""

target3b_alt = """            renderActiveSessionMessages();
        });"""

replacement3b_alt = """            renderActiveSessionMessages();
            setInputState(false);
        });"""

if target1 in content and target2 in content and target3a in content:
    content = content.replace(target1, replacement1)
    content = content.replace(target2, replacement2)
    content = content.replace(target3a, replacement3a)
    content = content.replace(target3b_alt, replacement3b_alt)
    open(path, 'w', encoding='utf-8').write(content)
    print("SUCCESS")
else:
    print("FAIL_TARGET_NOT_FOUND")
