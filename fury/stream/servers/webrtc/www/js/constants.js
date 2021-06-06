export const urlServer = `${location.protocol}//${location.hostname}${
  location.port ? ":" + location.port : ""
}/`;
export const videoEl = document.getElementById("video");

export const urlParams = new URLSearchParams(window.location.search);

export const weelInterval =
  urlParams.get("weel_interval") === null
    ? 30
    : parseFloat(urlParams.get("weel_interval"));

export const mouseInterval =
  urlParams.get("mouse_interval") === null
    ? 80
    : parseFloat(urlParams.get("mouse_interval"));

