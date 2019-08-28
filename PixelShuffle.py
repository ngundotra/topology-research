"""
PixelShuffle.py
--------------------
Scriptify the important things.

get_data --> returns Cifar DataBunch of a certain size, batchsize, and possibly shuffled
train_cycles -> trains an arch/model in cycles, generates the dataBunch from args
GlobalShuffle --- represents permutation info
"""
from fastai.vision import *
import torch
import torchvision.transforms

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (torch.FloatTensor([ 0.4914 ,  0.48216,  0.44653]), torch.FloatTensor([ 0.24703,  0.24349,  0.26159]))

def get_data(img_size, batchsize, with_shuffle=False):
    if with_shuffle:
        tfm = GlobalShuffle(img_size, img_size).get_tfm()
        data = ImageDataBunch.from_folder('cifar10/', size=img_size, bs=batchsize, ds_tfms=([tfm], [tfm]))
    else:
        data = ImageDataBunch.from_folder('cifar10/', size=img_size, bs=batchsize)
    return data.normalize(stats)

def get_data_with_tfm(img_size, batchsize):
    shuffleObj = GlobalShuffle(img_size, img_size)
    tfm = shuffleObj.get_tfm()
    data = ImageDataBunch.from_folder('cifar10/', size=img_size, bs=batchsize, ds_tfms=([tfm], [tfm]))
    return data.normalize(stats), shuffleObj

def train_cycles(model, sizes, batchsizes, num, learnertype, with_shuffle=False):
    if len(sizes) != len(batchsizes):
        raise ValueError("# of image sizes must match number of batchsizes")
    choices_arr = ["cnn", "learner"]
    if learnertype.lower() not in choices_arr:
        raise ValueError(f"Expected learner type to be one of {choices_arr}, got {learnertype}.")
    print(f"{num * len(sizes)} cycles will be trained. {int(len(sizes)*num*(num+1)/2)} total epochs will be run.")
    for k, (sz, bs) in enumerate(zip(sizes, batchsizes)):
        print(f"[{k}/{len(sizes)}] Size: {sz}, batchsize: {bs}")
        data = get_data(sz, bs, with_shuffle)
        if learnertype.lower() == "cnn":
            learner = cnn_learner(data, model, metrics=[accuracy])
        elif learnertype.lower() == "learner":
            learner = Learner(data, model, metrics=[accuracy])
        for cycle_len in range(num):
            learner.fit_one_cycle(cycle_len+1)
    return learner

class GlobalShuffle:
    def __init__(self, numRows, numCols):
        self.numRows = numRows
        self.numCols = numCols
        self.permuteVector = np.random.permutation(numRows * numCols)
        self.unpermuteVector = np.empty(numRows * numCols, self.permuteVector.dtype)
        self.unpermuteVector[self.permuteVector] = np.arange(numRows * numCols)
        
    def permuteArr(self, imgArr):
        """Permutes RGB rank 3 image"""
        return self._permuteArr(imgArr, self.permuteVector)
    
    def unPermuteArr(self, imgArr):
        """Unpermutes RGB rank 3 image"""
        return self._permuteArr(imgArr, self.unpermuteVector)
    
    def permuteTensor(self, imgTensor):
        return self._permuteTensor(imgTensor, self.permuteVector)
    
    def unPermuteTensor(self, imgTensor):
        return self._permuteTensor(imgTensor, self.unpermuteVector)
    
    def _permuteTensor(self, imgTensor, givenPermuteVector):
        """Permutes RGB rank 3 image torch tensor
        Assumes dimensions: [channel, height, width]
        Does not permute channel"""
        imgDim = list(imgTensor.shape)
        if len(imgDim) != 3:
            raise ValueError(f"Expected to have tensor with 3 dimensions, got {len(imgDim)} dimensions instead.")
        flatImgTensor = imgTensor.permute(1,2,0).reshape((-1, imgDim[0]))
        permutedImgTensor = flatImgTensor[givenPermuteVector]
        permutedImgTensor = permutedImgTensor.reshape(imgDim[1], imgDim[2], imgDim[0]).permute(2, 0, 1)
        return permutedImgTensor
    
    def createUnPermuteData(self):
        data = torch.zeros((self.numRows * self.numCols, self.numRows * self.numCols))
        for i, idx in enumerate(self.unpermuteVector):
            data[idx][i] = 1.0
        return data
    
    def _permuteArr(self, imgArr, permuteVector):
        """Internal method to simplify permutation"""
        flatImgArr = np.reshape(imgArr, (self.numRows * self.numCols, -1))
        flatPermutedArr = flatImgArr[permuteVector]
        return flatPermutedArr.reshape((self.numRows, self.numCols, -1))
    
    def get_tfm(self):
        shuffle_tfm = TfmPixel(lambda x: self.permuteTensor(x), order=1)
        return RandTransform(shuffle_tfm, {}, is_random=False)
    
def run_tests():
    """Runs torch permute tests"""
    ones_matrix = torch.Tensor(list(range(48)))
    # Make into normal image
    ones_matrix = ones_matrix.reshape((4, 4, 3))
    # Pytorch image style
    ones_torch_style = ones_matrix.permute(2, 0, 1)
    
    test_shuffle = GlobalShuffle(4, 4)
    shuffled_ones_torch_style = test_shuffle.permuteTensor(ones_torch_style)
    unshuffled_ones_torch_style = test_shuffle.unPermuteTensor(shuffled_ones_torch_style)
    
    if not (unshuffled_ones_torch_style == ones_torch_style).all():
        raise ValueError("UnPermuteTensor did not successfully unpermute the tensor.")
    
    unshuffled_ones_matrix = unshuffled_ones_torch_style.permute(1, 2, 0)
    if not (unshuffled_ones_matrix == ones_matrix).all():
        raise ValueError("Permutig the unshuffled matrix did not bring back to expected form.")
    
    return True
    
if __name__ == '__main__':
    if not run_tests():
        print("--------------------TESTS FAILED------------------------")
    else:
        print("--------------------TESTS PASSED------------------------")