var pc = null;
var bandwidth = 5;

function removeBandwidthRestriction(sdp) {
  return sdp.replace(/b=AS:.*\r\n/, "").replace(/b=TIAS:.*\r\n/, "");
}

function setMediaBitrates(sdp) {
  return setMediaBitrate(setMediaBitrate(sdp, "video", 5), "audio", 50);
}

function setMediaBitrate(sdp, media, bitrate) {
  var lines = sdp.split("\n");
  var line = -1;
  for (var i = 0; i < lines.length; i++) {
    if (lines[i].indexOf("m=" + media) === 0) {
      line = i;
      break;
    }
  }
  if (line === -1) {
    // console.debug("Could not find the m line for", media);
    return sdp;
  }
//   console.debug("Found the m line for", media, "at line", line);

  // Pass the m line
  line++;

  // Skip i and c lines
  while (lines[line].indexOf("i=") === 0 || lines[line].indexOf("c=") === 0) {
    line++;
  }

  // If we're on a b line, replace it
  if (lines[line].indexOf("b") === 0) {
    // console.debug("Replaced b line at line", line);
    lines[line] = "b=AS:" + bitrate;
    return lines.join("\n");
  }

  // Add a new b line
//   console.debug("Adding new b line before line", line);
  var newLines = lines.slice(0, line);
  newLines.push("b=AS:" + bitrate);
  newLines = newLines.concat(lines.slice(line, lines.length));
  return newLines.join("\n");
}
// videoBandwidth = 10;
function updateBandwidthRestriction(sdp, bandwidth) {
  let modifier = "AS";
  if (adapter.browserDetails.browser === "firefox") {
    bandwidth = (bandwidth >>> 0) * 1000;
    modifier = "TIAS";
  }
  if (sdp.indexOf("b=" + modifier + ":") === -1) {
    // insert b= after c= line.
    sdp = sdp.replace(
      /c=IN (.*)\r\n/,
      "c=IN $1\r\nb=" + modifier + ":" + bandwidth + "\r\n"
    );
  } else {
    sdp = sdp.replace(
      new RegExp("b=" + modifier + ":.*\r\n"),
      "b=" + modifier + ":" + bandwidth + "\r\n"
    );
  }
  return sdp;
}

function negotiate() {
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  return pc
    .createOffer()
    .then(function (offer) {
      let offerDescription = pc.setLocalDescription(offer);
      offer.sdp = setMediaBitrates(offer.sdp);
      return offerDescription;
    })
    .then(function () {
      // wait for ICE gathering to complete
      return new Promise(function (resolve) {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          function checkState() {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", checkState);
              resolve();
            }
          }
          pc.addEventListener("icegatheringstatechange", checkState);
        }
      });
    })
    .then(function () {
      var offer = pc.localDescription;
      return fetch("/offer", {
        body: JSON.stringify({
          sdp: setMediaBitrates(offer.sdp),
          type: offer.type,
        }),
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    })
    .then(function (response) {
      return response.json();
    })
    .then(function (answer) {
      // updateBandwidthRestriction(pc1.remoteDescription.sdp, bandwidth)
      console.log(answer);

      // const desc = {
      //     type: answer.type,
      //     sdp: updateBandwidthRestriction(answer.sdp, bandwidth)
      //   };
      answer.sdp = setMediaBitrates(answer.sdp);
      return pc.setRemoteDescription(answer);
    })
    .catch(function (e) {
      alert(e);
    });
}

export const startWebRTC = () => {
  var config = {
    // sdpSemantics: 'unified-plan'
  };

  //config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];

  config.iceServers = [{ urls: ["stun:stun.l.google.com:19302"] }];
  // if (document.getElementById('use-stun').checked) {
  //     config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
  // }

  pc = new RTCPeerConnection(config);

  // connect audio / video
  pc.addEventListener("track", function (evt) {
    if (evt.track.kind == "video") {
      document.getElementById("video").srcObject = evt.streams[0];
    } else {
      document.getElementById("audio").srcObject = evt.streams[0];
    }
  });

  document.getElementById('startBtn').style.display = 'none';
  negotiate();
  //document.getElementById('stop').style.display = 'inline-block';

  //     if ((adapter.browserDetails.browser === 'chrome' ||
  //        adapter.browserDetails.browser === 'safari' ||
  //        (adapter.browserDetails.browser === 'firefox' &&
  //         adapter.browserDetails.version >= 64)) &&
  //       'RTCRtpSender' in window &&
  //       'setParameters' in window.RTCRtpSender.prototype) {
  //     console.log("OKOKOKOKOKOKO")
  //     const sender = pc.getSenders()[0];
  //     const parameters = sender.getParameters();
  //     if (!parameters.encodings) {
  //       parameters.encodings = [{}];
  //     }
  //     if (bandwidth === 'unlimited') {
  //       delete parameters.encodings[0].maxBitrate;
  //     } else {
  //       parameters.encodings[0].maxBitrate = bandwidth * 1000;
  //     }
  //     sender.setParameters(parameters)
  //         .then(() => {
  //           bandwidthSelector.disabled = false;
  //         })
  //         .catch(e => console.error(e));
  //     return;
  //   }
}

function stop() {
  //document.getElementById('stop').style.display = 'none';

  // close peer connection
  setTimeout(function () {
    pc.close();
  }, 500);
}