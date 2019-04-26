import keras


def save_model(model, name):
    print('Saving Model')
    model_json = model.to_json()
    with open("models\\" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models\\" + name + ".h5")


def load_model(name):
    print('Saving Model')
    json_file = open("models\\" + name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("models\\" + name + ".h5")
    return model
