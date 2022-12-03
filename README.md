# flower-generation

The writeup of this project is Project_Writeup.pdf.

A significant amount of inspiration is drawn from the works referenced in the writeup, and especially from the [original IM-Net implementation](https://github.com/czq142857/IM-NET-pytorch).

The ProceduralGeneration file contains all the C# files that I wrote to generate flowers. Note that they require a Unity project to be run. The packages needed to reproduce this project are included. To reproduce the project, create a new Unity 3D project, and click on Assets -> Import Package -> Custom Package and then click on the flowersunity.package file.

The NetworkGeneration file contains all of the python files that were used to train the networks. The run.sh file contains all of the commands needed to train everything from scratch. However, a dataset must be created by running [binvox](https://www.patrickmin.com/binvox/) on a folder of generated flowers.



