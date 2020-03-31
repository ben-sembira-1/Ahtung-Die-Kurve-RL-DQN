CURVE_FEVER_URL = 'http://forum.curvefever.com/play.html'
FLASH_OPTIONS_URL = 'chrome://settings/content/flash'
FLASH_OPTIONS_ELEMENT_XPATH = '//*[@id="labelWrapper"]/div[1]'

X_PATH_START_STR = "//*[contains(text(''), '{0}')]"

CHROME_PREFERENCES = {
    "RunAllFlashInAllowMode":"true",
    "profile.default_content_settings.state.flash": 0,
    "profile.plugins.flashBlock.enabled": 1,
    "profile.default_content_setting_values.plugins": 1,
    "DefaultPluginsSetting": 1,
    "AllowOutdatedPlugins":1,
    "profile.content_settings.plugin_whitelist.adobe-flash-player": 1,
    "profile.content_settings.exceptions.plugins.*,*.per_resource.adobe-flash-player": 1,
    "PluginsAllowedForUrls": CURVE_FEVER_URL,
    "--allow-running-insecure-content": 1,
    "--allow-insecure-websocket-from-https-origin": 1,
    "allow-outdated-plugins" :1
}
