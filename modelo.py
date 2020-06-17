from PIL import Image
import tensorflow as tf
import numpy as np

pic_number = 0

class Model:

    def __init__(self, model_file, dict_file):
        with open(dict_file, 'r') as f:
            self.labels = [line.strip().replace('_', ' ') for line in f.readlines()]
        self.interpreter = tf.lite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def classify(self, file, maxResults, min_confidence):

        print("Confidence level %f" % min_confidence);

        with Image.open(file).resize((self.width, self.height)) as img:
            input_data = np.expand_dims(img, axis=0)
            if self.floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            results = np.squeeze(output_data)
            top_categories = results.argsort()[::-1]
            if maxResults != None:
                top_categories = top_categories[:maxResults]
            print("==> %s <==" % file)

            final_results = []

            for i in top_categories:
                if self.floating_model:
                    r = float(results[i])
                else:
                    r = float(results[i] / 255.0)
                if min_confidence != None and r < min_confidence:
                    break

                global pic_number

                res = "{:6.2f}%: {}: %d".format(r*100, self.labels[i], pic_number)
                pic_number += 1
                #print('{:06.4f}: {}'.format(r, self.labels[i]))
                #print(res)
                final_results.append(res)


            return final_results