from nsfw_detector import predict
model = predict.load_model('mobilenet_v2_140_224')

# Predict single image
result = predict.classify(model, '../../input/nsfw/data/porn')

#print(result)
for r in result:
    print(r)
    print(result[r])