![tonic](tonic-logo-padded.png)
Tonic provides spike-based datasets and a pipeline of data augmentation methods.

## Install
```bash
pip install tonic
```

## Quickstart
```python
import tonic
import tonic.transforms as transforms

transform = transforms.Compose([transforms.TimeJitter(variance=10),
                                transforms.FlipLR(flip_probability=0.5),
                                transforms.ToTimesurface(surface_dimensions=(7,7), tau=5e3),])

testset = tonic.datasets.NMNIST(save_to='./data',
                                                  train=False,
                                                  transform=transform)

testloader = tonic.datasets.Dataloader(testset, shuffle=True)

for surfaces, target in iter(testloader):
    print("{} surfaces for target {}".format(len(surfaces), target))
```

## Documentation
To see a list of all transforms and their possible parameters, it is necessary to build documentation locally. Just run the following commands to do that:
```bash
cd docs
make html
firefox _build/html/index.html
```

## Possible data sets (asterix marks currently supported in this package)
- [MVSEC](https://daniilidis-group.github.io/mvsec/)
- [NMNIST](https://www.garrickorchard.com/datasets/n-mnist) (\*)
- [ASL-DVS](https://github.com/PIX2NVS/NVS2Graph)
- [NCARS](https://www.prophesee.ai/dataset-n-cars/)(\*)
- [N-CALTECH 101](https://www.garrickorchard.com/datasets/n-caltech101)(\*)
- [POKER-DVS](http://www2.imse-cnm.csic.es/caviar/POKERDVS.html) (\*)
- [IBM gestures](http://www.research.ibm.com/dvsgesture/) (\*)
- [TI Digits](https://catalog.ldc.upenn.edu/LDC93S10)
- [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)

## Contribute

#### Install pre-commit

Please use the [black formatter](https://black.readthedocs.io/en/stable/) as a pre-commit hook. You can easily install it as follows:
```
pip install pre-commit
pre-commit install
```
When you use ```git add``` you add files to the current commit, then when you run ```git commit``` the black formatter will run BEFORE the commit itself. If it fails the check, the black formatter will format the file and then present it to you to add it into your commit. Simply run ```git add``` on those files again and do the remainder of the commit as normal.

#### Run tests

To run the tests, from the root directory of the repo

```
python -m pytest
```
