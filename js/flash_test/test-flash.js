function getFlashState() {
    var flashState = 1;
    try {
        var ao = new ActiveXObject('ShockwaveFlash.ShockwaveFlash');
        if (ao) {
            flashState = 0;
        }
    } catch (e) {
        var mime = navigator.mimeTypes ? navigator.mimeTypes['application/x-shockwave-flash'] : undefined;
        var xsf = mime != undefined;
        var xsfEnabled = xsf && mime.enabledPlugin;
        flashState = xsfEnabled ? 0 : xsf ? 2 : 1;
        console.log(e);
    }
    return flashState;
}
function testFlash(succesRedirect, errorOutputID_1, errorOutputID_2, args) {
    var flashState = getFlashState();
    if (flashState == 0) {
        if (succesRedirect) {
            window.location.href = succesRedirect;
        }
        return 0;
    }
    if (flashState == 1 && errorOutputID_1)
        document.getElementById(errorOutputID_1).style.display = "block";
    if (flashState == 2 && errorOutputID_2)
        document.getElementById(errorOutputID_2).style.display = "block";
    if (flashState != 0 && args && args.length > 0) {
        for (var i = 0; i < args.length; i++) {
            var element = document.getElementById(args[i]);
            if (element != null) {
                element.className += " disabled";
                element.title = (flashState == 1 ? "Install" : "Enable") + " Flash to play";
            } else {
                console.warn("Could not get element by ID: " + args[i]);
            }
        }
    }
    document.getElementById("main").style.display = "inherit";
    return flashState;
}
