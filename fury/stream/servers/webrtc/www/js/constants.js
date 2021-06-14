export const urlServer = `${location.protocol}//${location.hostname}${
  location.port ? ":" + location.port : ""
}/`;

export const urlParams = new URLSearchParams(window.location.search);

export const weelInterval =
  urlParams.get("weel_interval") === null
    ? 30
    : parseFloat(urlParams.get("weel_interval"));

export const encoding =
  urlParams.get("encoding") === null
    ? "webrtc"
    : urlParams.get("encoding");
console.log('enconding', encoding)
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
