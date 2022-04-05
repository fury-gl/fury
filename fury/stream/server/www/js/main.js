import { startWebRTC } from "/js/webrtc.js";
import {
  urlParams,
  encoding,
} from "/js/constants.js";
import {
  addInteraction
} from "/js/interaction.js";

function getVideoDOMEL(encoding) {
  let videoElId;
  switch (encoding) {
    case "webrtc":
      videoElId = "video";
      break;
    case "mjpeg":
      videoElId = "videoMJPEG";
      break;
    default:
      videoElId = "video";
  }
  return document.getElementById(videoElId)
}
document.addEventListener("DOMContentLoaded", (event) => {
  const videoEl = getVideoDOMEL(encoding)
  if (encoding === "webrtc") {
    document.getElementById("startBtn").className = ''
    document.getElementById("startBtn").addEventListener(
      "click", (e) => 
        startWebRTC()
    )
  } else if (encoding === 'mjpeg') {
    document.getElementById("startBtn").className = 'hidden'
    videoEl.src = `/video/mjpeg`
  }
  const interaction = urlParams.get("interaction");
  const runningOnIframe = urlParams.get("iframe");
  const backgroundColor = urlParams.get("background");

  if (interaction === null || interaction == 1) addInteraction(videoEl);
  const videoClass =
    runningOnIframe == null || runningOnIframe == 0
      ? "videoNormalMode"
      : "videoIframeMode";
  videoEl.className = videoClass;
  if (backgroundColor != null)
    document.body.style.backgroundColor = backgroundColor;
});


