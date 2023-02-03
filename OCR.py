import cv2
import pytesseract
import os


class OCR:
    def __init__(self):
        pass

    def getText(self, path):
        parses = {'name': "",
                  'place': "",
                  "arabicId": "",
                  "engId": ""
                  }

        for filename in os.listdir(path):
            name = filename.split(".")[0]
            if name in parses:
                filepath = f"{path}/{filename}"
                img = cv2.imread(filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

                # Adding custom options
                if name == "name" or name == "place":
                    custom_config = r'--psm 6 -l ara-Amiri'

                if name == "place":
                    custom_config = r'--psm 3 -l ara-Amiri'

                if name == "engId":
                    custom_config = r'--psm 6'

                if name == "arabicId":
                    custom_config = r'--psm 7 -l ara-Amiri'

                s = pytesseract.image_to_string(img, config=custom_config)

                print(s)

                parses[name] = s

        return parses
