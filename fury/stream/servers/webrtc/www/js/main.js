import { startWebRTC } from "/js/webrtc.js";
import {
  urlServer,
  weelInterval,
  mouseInterval,
  videoEl,
  urlParams,
} from "/js/constants.js";

let addrInteraction =
  urlParams.get("interaction_addr") === null
    ? `${location.hostname}:${location.port}`
    : urlParams.get("interaction_addr");

const urlInteraction = `${location.protocol}://${addrInteraction}/`;
const clientSocket = new WebSocket(`ws://${addrInteraction}/ws`);

document.addEventListener("DOMContentLoaded", (event) => {
  document.getElementById("startBtn").addEventListener("click", (e) => {
    startWebRTC();
  });
  const interaction = urlParams.get("interaction");
  const runningOnIframe = urlParams.get("iframe");
  const backgroundColor = urlParams.get("background");

  console.info("address interaction server", urlInteraction);
  if (interaction === null || interaction == 1)
    addInteraction(videoEl);
  const videoClass =
    runningOnIframe == null || runningOnIframe == 0
      ? "videoNormalMode"
      : "videoIframeMode";
  videoEl.className = videoClass;
  if (backgroundColor != null)
    document.body.style.backgroundColor = backgroundColor;
});

function addInteraction(videoEl) {
  let mouseLeftReleased = true;
  let mouseOutVideo = false;
  let mouseX = 0;
  let mouseY = 0;
  let ctrlKey = 0;
  let shiftKey = 0;
  let enableCallWheel = true;
  let enableCallMove = true;
  let currentWheelEventTotalDeltaY = 0;

  videoEl.addEventListener("wheel", (event) => {
    event.preventDefault(); 
    currentWheelEventTotalDeltaY += event.deltaY;
    if (!enableCallWheel) return;

    const data = {
      type: "weel",
      deltaY: currentWheelEventTotalDeltaY,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);
    currentWheelEventTotalDeltaY = 0;
    clientSocket.send(dataJson);
    // .then((response) => response.json())
    // .then((data) => {
    //   // console.log("Success:", data)
    // })
    // .catch((error) => {
    //   console.error("Error:", error);
    // });
    enableCallWheel = false;
    setTimeout(() => (enableCallWheel = true), weelInterval);
  });

  videoEl.addEventListener("mousemove", (event) => {
    if (!enableCallMove) return;
    mouseOutVideo = false;
    ctrlKey = event.ctrlKey ? 1 : 0;
    shiftKey = event.shiftKey ? 1 : 0;
    const width = videoEl.offsetWidth;
    const height = videoEl.offsetHeight;
    mouseX = event.offsetX / width;
    mouseY = event.offsetY / height;
    mouseMoveCallback();
    enableCallMove = false;
    setTimeout(() => (enableCallMove = true), mouseInterval);
  });

  videoEl.addEventListener("mouseleave", (e) => (mouseOutVideo = false));
  // videoEl.addEventListener("mousemove", (e) => {
  //   if (mouseLeftReleased) return
  //   mouseX = e.offsetX
  //   mouseY = e.offsetY
  //   const width = videoEl.offsetWidth
  //   const height = videoEl.offsetHeight
  //   let x = mouseX/height//width
  //   let y = mouseY/width///height
  //   const data = { x: x, y: y }
  //   console.log(x*100, y*100)
  //   fetch("http://localhost:8000/mouse_move", {
  //     method: "POST", // or 'PUT'
  //     headers: {
  //       "Content-Type": "application/json",
  //     },
  //     body: JSON.stringify(data),
  //   })
  //     .then((response) => response.json())
  //     .then((data) => {
  //       // console.log("Success:", data)
  //     })
  //     .catch((error) => {
  //       console.error("Error:", error)
  //     })
  // })

  const mouseMoveCallback = () => {
    if (mouseLeftReleased || mouseOutVideo) return;
    const data = {
      type: "mouseMove",
      x: mouseX,
      y: mouseY,
      ctrlKey: ctrlKey,
      shiftKey: shiftKey,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);
    clientSocket.send(dataJson);
  };
  // const timerMouseMove = setInterval(mouseMoveCallback, mouseInterval);

  videoEl.addEventListener("mousedown", (e) => {
    //e.preventDefault()
    mouseLeftReleased = false;
    ctrlKey = e.ctrlKey ? 1 : 0;
    shiftKey = e.shiftKey ? 1 : 0;
    const mouseButton = e.button;
    const data = {
      type: "mouseLeftClick",
      on: 1,
      mouseButton: mouseButton,
      x: mouseX,
      y: mouseY,
      ctrlKey: ctrlKey,
      shiftKey: shiftKey,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);

    clientSocket.send(dataJson);
  });

  window.addEventListener("mouseup", (e) => {
    mouseLeftReleased = true;
    ctrlKey = e.ctrlKey ? 1 : 0;
    shiftKey = e.shiftKey ? 1 : 0;
    const mouseButton = e.button;
    const data = {
      type: "mouseLeftClick",
      on: 0,
      x: mouseX,
      y: mouseY,
      mouseButton: mouseButton,
      ctrlKey: ctrlKey,
      shiftKey: shiftKey,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);

    clientSocket.send(dataJson);
  });

  // document.addEventListener("keydown", function (event) {
  //   ctrlKey = event.ctrlKey ? 1:0
  //   shiftKey =event.shiftKey ? 1:0
  // if (!event.ctrlKey) {

  //   fetch("http://localhost:8000/ctrl_key", {
  //     method: "POST", // or 'PUT'
  //     headers: {
  //       "Content-Type": "application/json",
  //     },
  //     body: JSON.stringify({'on': 1}),
  //   })
  //     .then((response) => response.json())
  //     .then((data) => {
  //       // console.log("Success:", data)
  //     })
  //     .catch((error) => {
  //       console.error("Error:", error)
  //     })
  // }
  // })
  // document.addEventListener("keyup", function (event) {
  //   ctrlKey = event.ctrlKey ? 1:0
  //   shiftKey =event.shiftKey ? 1:0

  // if (event.ctrlKey) {
  //   console.log('ctrl up')
  //   fetch("http://localhost:8000/ctrl_key", {
  //     method: "POST", // or 'PUT'
  //     headers: {
  //       "Content-Type": "application/json",
  //     },
  //     body: JSON.stringify({'on': 0}),
  //   })
  //     .then((response) => response.json())
  //     .then((data) => {
  //       // console.log("Success:", data)
  //     })
  //     .catch((error) => {
  //       console.error("Error:", error)
  //     })
  // }
  // })

  // document.getElementById('startBtn').addEventListener('wheel', (event)=>
  // {
  //     console.log(event.deltaY)
  // })
}
