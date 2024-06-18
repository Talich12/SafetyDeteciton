from ultralytics import YOLO


model = YOLO("best.pt") 

# Predict with the model
results = model.predict("JapanPPE.mp4", save=True, conf=0.25)  # predict on an image


#for result in results:
  #  result.show()