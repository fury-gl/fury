import { startWebRTC } from "/js/webrtc.js";

document.addEventListener("DOMContentLoaded", (event) => {
  document.getElementById("startBtn").addEventListener("click", (e) => {
    startWebRTC();
  });
});

window.addEventListener("wheel", (event) => {
  const data = { deltaY: event.deltaY };

  fetch("http://localhost:8000/mouse_weel", {
    method: "POST", // or 'PUT'
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Success:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
});
// document.getElementById('startBtn').addEventListener('wheel', (event)=>
// {
//     console.log(event.deltaY)
// })
