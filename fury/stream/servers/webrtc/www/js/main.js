import { startWebRTC } from "/js/webrtc.js"
import { millisecMouseMove } from "/js/constants.js"

let mouseLeftReleased = true
let mouseOutVideo = false
let mouseX = 0
let mouseY = 0
let ctrlKey = 0
let shiftKey = 0
const videoEl = document.getElementById("video")

document.addEventListener("DOMContentLoaded", (event) => {
  document.getElementById("startBtn").addEventListener("click", (e) => {
    startWebRTC()
  })
})

videoEl.addEventListener("wheel", (event) => {
  const data = { deltaY: event.deltaY }

  fetch("http://localhost:8000/mouse_weel", {
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
      console.error("Error:", error)
    })
})


videoEl.addEventListener("mousemove", (event) => {
  mouseOutVideo = false
  ctrlKey = event.ctrlKey ? 1:0
  shiftKey = event.shiftKey ? 1:0
  const width = videoEl.offsetWidth
  const height = videoEl.offsetHeight
  mouseX = event.offsetX/width
  mouseY = event.offsetY/height
  
})

videoEl.addEventListener("mouseleave", (e) => {
  mouseOutVideo = false
})
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

const mouseMoveCallback = ()=>{
    if (mouseLeftReleased || mouseOutVideo) return
    console.log(ctrlKey, shiftKey)
    const data = {x:mouseX, y:mouseY, ctrlKey:ctrlKey, shiftKey:shiftKey}
    fetch("http://localhost:8000/mouse_move", {
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
      console.error("Error:", error)
    })
}
const timerMouseMove = setInterval(mouseMoveCallback, millisecMouseMove);


videoEl.addEventListener("mousedown", (e) => {
  mouseLeftReleased = false
})

window.addEventListener("mouseup", (e) => {
  mouseLeftReleased = true
  fetch("http://localhost:8000/mouse_left_click", {
    method: "POST", // or 'PUT'
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({'on': 0}),
  })
    .then((response) => response.json())
    .then((data) => {
      // console.log("Success:", data)
    })
    .catch((error) => {
      console.error("Error:", error)
    })
})

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
