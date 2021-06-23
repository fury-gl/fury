export const urlParams = new URLSearchParams(window.location.search);

export const weelInterval =
  urlParams.get("weel_interval") === null
    ? 30
    : parseFloat(urlParams.get("weel_interval"));

export const encoding =
  urlParams.get("encoding") === null
    ? "webrtc"
    : urlParams.get("encoding");

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
export const videoEl = getVideoDOMEL(encoding)
export const mouseInterval =
  urlParams.get("mouse_interval") === null
    ? 80
    : parseFloat(urlParams.get("mouse_interval"));


const socketProtocol =
  location.protocol === 'https:'
    ? 'wss'
    : 'ws';

export const socketAddr = `${socketProtocol}://${location.hostname}:${location.port}/ws`
export const httpURL = `${location.protocol}://${location.hostname}:${location.port}/`