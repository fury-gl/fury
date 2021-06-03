import { startWebRTC } from "/js/webrtc.js";
import {
  millisecMouseMove,
  urlServer,
  interactionInterval,
  videoEl,
  urlParams,
} from "/js/constants.js";

document.addEventListener("DOMContentLoaded", (event) => {
  document.getElementById("startBtn").addEventListener("click", (e) => {
    startWebRTC();
  });
  const interaction = urlParams.get("interaction");
  const runningOnIframe = urlParams.get("iframe");
  const backgroundColor = urlParams.get('background')
  if (interaction === null || interaction == 1) addInteraction(videoEl);
  const videoClass =
    runningOnIframe == null || runningOnIframe == 1
      ? "videoIframeMode"
      : "videoNormalMode";
  videoEl.className = videoClass
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
  let currentWheelEventTotalDeltaY = 0;
  videoEl.addEventListener("wheel", (event) => {
    currentWheelEventTotalDeltaY += event.deltaY;
    if (!enableCallWheel) return;
    const data = { deltaY: currentWheelEventTotalDeltaY };
    currentWheelEventTotalDeltaY = 0;

    fetch(`${urlServer}mouse_weel`, {
      method: "POST", // or 'PUT'
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    // .then((response) => response.json())
    // .then((data) => {
    //   // console.log("Success:", data)
    // })
    // .catch((error) => {
    //   console.error("Error:", error);
    // });
    enableCallWheel = false;
    setTimeout(() => (enableCallWheel = true), interactionInterval);
  });

  videoEl.addEventListener("mousemove", (event) => {
    mouseOutVideo = false;
    ctrlKey = event.ctrlKey ? 1 : 0;
    shiftKey = event.shiftKey ? 1 : 0;
    const width = videoEl.offsetWidth;
    const height = videoEl.offsetHeight;
    mouseX = event.offsetX / width;
    mouseY = event.offsetY / height;
  });

  videoEl.addEventListener("mouseleave", (e) => {
    mouseOutVideo = false;
  });
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
    const data = { x: mouseX, y: mouseY, ctrlKey: ctrlKey, shiftKey: shiftKey };
    fetch(`${urlServer}mouse_move`, {
      method: "POST", // or 'PUT'
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        // console.log("Success:", data)
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };
  const timerMouseMove = setInterval(mouseMoveCallback, millisecMouseMove);

  videoEl.addEventListener("mousedown", (e) => {
    mouseLeftReleased = false;
  });

  window.addEventListener("mouseup", (e) => {
    mouseLeftReleased = true;
    fetch(`${urlServer}mouse_left_click`, {
      method: "POST", // or 'PUT'
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ on: 0 }),
    })
      .then((response) => response.json())
      .then((data) => {
        // console.log("Success:", data)
      })
      .catch((error) => {
        console.error("Error:", error);
      });
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
