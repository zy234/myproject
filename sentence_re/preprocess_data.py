
FILE_PATH = "/data/zhaoy/myProject/ditil_for_semevel/data"
train_data_path = "/log_SEtrain.csv"
test_data_path = "/log_SEtest.csv"

# -----------------------------------------------------------------------------
train_data = []
train_target = []
# with open(FILE_PATH + train_data_path, "r") as f:
with open(FILE_PATH + test_data_path, "r") as f:
    while True:
        line1 = f.readline().strip()
        if not line1: break
        line2 = f.readline().strip().split()

        train_data.append(line1)
        if "B-Cause" in line2:
            train_target.append("1") # 1 means the sentence have ce relation
        else:
            train_target.append("0")

# with open("./data/train.tsv", "w") as f:
#     for (sentence, target) in zip(train_data, train_target):
#         f.write(sentence + "\t" + target + "\n")
with open("./data/test.tsv", "w") as f:
    for (sentence, target) in zip(train_data, train_target):
        f.write(target + "\t" + sentence + "\n")
# --------------------------------------------------------------------------------------