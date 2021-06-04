export const urlServer = `${location.protocol}//${location.hostname}${
  location.port ? ":" + location.port : ""
}/`;
export const videoEl = document.getElementById("video");

export const urlParams = new URLSearchParams(window.location.search);

export const millisecMouseMove = 100;
export const interactionInterval = 100;