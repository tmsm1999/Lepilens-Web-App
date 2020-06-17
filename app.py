import os
import shutil
import modelo
from flask import Flask, render_template, request


debug_output=False
min_confidence = 0.05
max_results = 5
pic_number = 0

application = Flask(__name__)

model_file = application.root_path + "/model.tflite"
label_file = application.root_path + "/dict.txt"

model = modelo.Model(model_file, label_file)

application.config["IMAGE_STATIC"] = application.root_path + "/static"

@application.route("/")
def initHTML():
    return render_template('Model_WebPage.html')

@application.route("/classify_image", methods=["GET", "POST"])
def classify_image():

    confidence_slider = request.form["confidence_slider"]
    min_confidence = float(confidence_slider) / 100
    if debug_output:
      print(min_confidence)

    dictionary = {}

    if request.method == "POST":
        if request.files:
            images = request.files.getlist("image_to_classify")

            for image in images:
                image.save(os.path.join(application.config["IMAGE_STATIC"], image.filename))

                path_to_image = os.path.join(application.config["IMAGE_STATIC"], image.filename)
                res_list = model.classify(path_to_image, max_results, min_confidence)

                if debug_output:
                   print(len(res_list))

                dictionary[image.filename] = res_list
                if len(res_list) == 0:
                    dictionary[image.filename] = ["No results for the chosen confidence level"]
                if debug_output:
                  for elem in res_list:
                    print(elem)

    return render_template('Model_WebPage.html', species_list=dictionary)

if __name__ == "__main__":
    application.run()
