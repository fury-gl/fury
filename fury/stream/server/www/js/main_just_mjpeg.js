import {
  urlParams,
} from "/js/constants.js";
import {
  addInteraction
} from "/js/interaction.js";

document.addEventListener("DOMContentLoaded", (event) => {
  const videoEl = document.getElementById('videoMJPEG')
  videoEl.src = `/video/mjpeg`
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


