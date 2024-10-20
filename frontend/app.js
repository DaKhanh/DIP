document.getElementById('chat-form').addEventListener('submit', function (e) {
    e.preventDefault();

    // 获取用户输入的问题
    const userInput = document.getElementById('user-input').value;

    // 将用户问题添加到聊天历史中
    const chatHistory = document.getElementById('chat-history');
    const userMessage = document.createElement('li');
    userMessage.textContent = "You: " + userInput;
    chatHistory.appendChild(userMessage);

    // 清空输入框
    document.getElementById('user-input').value = '';

    // 发送请求到后端的 /ask API
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // 显示 AI 的回答
        const botMessage = document.createElement('li');
        botMessage.textContent = "GPT EdWiser: " + data.answer;
        chatHistory.appendChild(botMessage);

        // 自动滚动到最新消息
        chatHistory.scrollTop = chatHistory.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
