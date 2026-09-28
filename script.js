// const input = document.querySelector(".chat-input input");
// const button = document.querySelector(".chat-input button");
// const messages = document.querySelector(".chat-messages");

// button.addEventListener("click", function () {

//     const userText = input.value.trim().toLowerCase();

//     if (userText === "") {
//         return;
//     }

//     // Show user's message
//     const userMessage = document.createElement("div");
//     userMessage.classList.add("user-message");
//     userMessage.textContent = input.value;

//     messages.appendChild(userMessage);

//     // Create bot response
//     const botMessage = document.createElement("div");
//     botMessage.classList.add("bot-message");

//     if (userText.includes("admission")) {
//         botMessage.textContent =
//             "For admission information, please visit the college admission office or official website.";

//     } else if (userText.includes("fee") || userText.includes("fees")) {
//         botMessage.textContent =
//             "For information about college fees, please check the official fee structure.";

//     } else if (userText.includes("timing") || userText.includes("time")) {
//         botMessage.textContent =
//             "College timings are usually from 9 AM to 5 PM. Please check your college's official schedule.";

//     } else if (userText.includes("hello") || userText.includes("hi")) {
//         botMessage.textContent =
//             "Hello! 👋 How can I help you with your college-related questions?";

//     } else {
//         botMessage.textContent =
//             "I'm sorry, I don't have information about that yet. Please try asking about admissions, fees, or timings.";
//     }

//     messages.appendChild(botMessage);

//     input.value = "";
// });
// input.addEventListener("keypress", function(event) {
//     if (event.key === "Enter") {
//         button.click();
//     }
// });