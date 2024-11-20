import reward

model = reward.load("ImageReward-v1.0")
prompt = ""
rewards = model.score(prompt, ["image1 path", "image2 path", ...])
