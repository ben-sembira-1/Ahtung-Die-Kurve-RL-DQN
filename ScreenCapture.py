from selenium import webdriver

driver = webdriver.Chrome(executable_path='C:/bin/chromedriver.exe')
driver.get('http://forum.curvefever.com/play.html')
driver.save_screenshot("screenshot.png")
