$body = @{
    query = "What is the 5:1 ratio in workplace relationships"
    answer = "The 5:1 ratio refers to maintaining five positive comments for every one negative comment."
    contexts = @("Researchers have discovered that high-performing teams and people have a secret to their success. They use a ratio of five positive comments or responses for every negative one. This sets a constructive tone for working well with others. Negativity has a very powerful and of course, negative, impact on relationships. Researchers have found that it takes five positive responses to undo the damage of one negative comment.")
}

$json = $body | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8502/evaluate" -Method Post -Body $json -ContentType "application/json"
