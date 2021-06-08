from model import Mask_RCNN_Model

training = False

if training:
    categories = ['person', 'pizza']
    model = Mask_RCNN_Model("CW-person-pizza",pretrained=False,categories=categories)
    model.train(load_existing_weights=False, epochs=20)
    model.predict(img_path='output/pizza.jpg')
else:
    categories = ['person', 'pizza']
    model = Mask_RCNN_Model("CW-person-pizza",pretrained=False,categories=categories)
    model.train(load_existing_weights=True)
    model.predict(img_path='output/pizza.jpg')
