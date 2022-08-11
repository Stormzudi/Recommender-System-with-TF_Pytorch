import torch




def score(logit_scale,image_features, text_features):
    image_features = torch.Tensor(image_features)
    text_features = torch.Tensor(text_features)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_per_image = logit_scale * image_features @ text_features.t()
    probs = logit_per_image.softmax(dim=-1).cpu().detach().numpy()
    return probs


def getNextMaxItem(result, ignore_range):
    maxScore = None
    maxIndex = None
    for index, score in enumerate(result):
        if index in ignore_range:
            continue
        if maxScore is None:
            maxScore = score
            maxIndex = index
        if score > maxScore:
            maxScore = score
            maxIndex = index

    return maxScore, maxIndex


def getRange(maxIndex, maxScore, result, thod=0.1, ignore_range=None):
    leftIndex = maxIndex
    rightIndex = maxIndex
    has_ignore_range = ignore_range is not None

    for i in range(maxIndex):
        prev_index = maxIndex - 1 - i
        if has_ignore_range and prev_index in ignore_range:
            break
        if result[prev_index] >= maxScore - thod:
            leftIndex = prev_index
        else:
            break
    for i in range(maxIndex + 1, len(result)):
        if has_ignore_range and i in ignore_range:
            break
        if result[i] >= maxScore - thod:
            rightIndex = i
        else:
            break
    if (rightIndex - leftIndex) > 60:
        return getRange(maxIndex, maxScore, result, thod / 2, ignore_range)
    return leftIndex, max(rightIndex, leftIndex + 10)


def getMultiRange(result, thod=0.1, maxcount=5):
    ignore_range = []
    index_list = []
    for i in range(maxcount):
        maxScore, maxIndex = getNextMaxItem(result, ignore_range)
        if maxIndex is None:
            break
        leftIndex, rightIndex = getRange(maxIndex, maxScore, result, thod, ignore_range)
        index_list.append([leftIndex,rightIndex])
        ignore_range.extend(list(range(leftIndex, rightIndex + 1)))
    return index_list