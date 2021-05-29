import {startWebRTC} from '/js/webrtc.js'

document.addEventListener("DOMContentLoaded", (event) => {
  document.getElementById('startBtn').addEventListener('click', (e)=>{startWebRTC()})
});

window.addEventListener('wheel', (event)=>
{
    console.log(event.deltaY)
})