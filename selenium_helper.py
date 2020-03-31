from constants import *
from selenium.webdriver.common.keys import Keys

class ScreenCapture:

    def capture_game(self, driver):
        driver.get(CURVE_FEVER_URL)
        self.set_flash(driver)
        driver.save_screenshot("screenshot.png")

    def set_flash(self, driver):
        driver.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 't')
        driver.get(FLASH_OPTIONS_URL)
        btn = driver.find_elements_by_xpath(FLASH_OPTIONS_ELEMENT_XPATH)[0]
        btn.click()


class Navigate:

    def press_by_label(self, label, driver):
        button = driver.find_elements_by_xpath('')
        button[0].click()
