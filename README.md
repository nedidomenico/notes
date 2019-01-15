# VOLANS
The Volans platform includes an API and a set of tools for handling and working with large remote sensing data sets.  Highlights of the Volans capabilities include:
1. Simplified reading of satellite imagery using GDAL and NITRO
2. Integration of GDAL/NITRO with the image processing power of OpenCV
3. Implementation of common sensor models used to convert image coordinates to longitude and latitude
4. Integration of sensor models with JTS geometric data structures
5. Convenient data structures for working with tiles from large remote sensing images
6. Application of Convolutional Neural Networks (CNNs) to large remote sensing images
7. Scalable Automated Target Recognition on AWS
     * Apply CNNs to arbitrarily large imagey data set on S3
     * Use elastic scaling of multiple EC2 instances each with multiple GPUs
     * Write results as GeoTools Simple Features to GeoMesa
     * Visualize and vet results in Stealth or use for Multi-Int analytics``
     
## Example Usage
### Initial Setup
Note: volans needs to know which version of cuda is installed on the machine you are running. By default, volans will build with cuda-9.2. To build a different version, build with `-Dcuda.version=MY_VERSION`. e.g. `mvn -Dcuda.version=9.0 clean install`.

Volans will write required natives to the following location:
```
/tmp/username/volans/natives
```
Note: you can set a different directory by setting the -Djava.io.tmpdir flag

When running on df01 or udev, the following java options may be required:
```
-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS \
-Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.F2jLAPACK \
-Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.F2jARPACK
```

### Basic Image Processing
Open a CvGdalReader for a NITF image:
```
NativeLoader.init()  // loads binaries
val imgPath = new java.io.File("/net/synds1/volume1/projects/dragonfish/multiview-stereo")
val imgFile = new java.io.File(imgPath, "01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF")
val outPath = new java.io.File("/path/to/output")
val gdalReader = CvGdalReader(imgFile)
```
Display the number of columns (samples) and rows (lines):
```
println(s"Number of columns = ${gdalReader.nXCols}")
println(s"Number of rows = ${gdalReader.nYRows}")
println(s"Number of channels = ${gdalReader.channels}")
```
```
Number of columns = 43008
Number of rows = 43008
Number of channels = 1
```
Read a block of image data into OpenCV Mats:
```
val rect = new opencv_core.Rect(512, 512, 512, 512)
val imgMat: Map[Int, opencv_core.Mat] = gdalReader.read(rect)
```
A few comments about reading image data:
* Each channel is read into a separate Mat
* The Map[Int, Mat] is ordered by channel 1...N from shortest to longest wavelength (e.g. BGR).  See below for working with multi-spectral data
* Mat types are either opencv_core.CV_8U (byte) or opencv_core.CV_16U (unsigned int)

Display the size and type of the Mat for channel 1:
```
val matSize = imgMat(1).size()
println(s"Mat number of columns = ${matSize.width}")
println(s"Mat number of rows = ${matSize.height}")
println(s"Mat type = ${gdalReader.cvType}")
println(s"  opencv_core.CV8U = ${opencv_core.CV_8U}")
println(s"  opencv_core.CV16U = ${opencv_core.CV_16U}")
```
This data type is opencv_core_16U:
```
Mat number of columns = 512
Mat number of rows = 512
Mat type = 2
  opencv_core.CV8U = 0
  opencv_core.CV16U = 2

```
Display the minimum and maximum values for the mat and get the pixel value at (col, row) = (112, 442).  Note that the pixel value is cast to type Int:
```
val matMinMax = CvUtils.minMax(imgMat(1))
val pixVal = CvUtils.getMatValue[Int](imgMat(1), 112, 442)
println(s"Min mat pixel value = ${matMinMax._1}")
println(s"Max mat pixel value = ${matMinMax._2}")
println(s" Pixel value at (112, 442) = $pixVal")
```
```
Min mat pixel value = 169.0
Max mat pixel value = 2047.0
Pixel value at (112, 442) = 326
```
Write the image to a png file.  Note that the image data is 16-bits.  PNG supports 16-bits, but many viewers require 8-bits.  For easy viewing, scale and convert the mat to 8-bits:
```
val pngFile = new java.io.File(imgPath, "pngExample.png").toString()
val scaledMat = CvUtils.scaleMatMinMax(imgMat(1))
CvUtils.writeMat(scaledMat, pngFile)
```
<img src="/uploads/f383ff9b85cc6b3ab5339d02c02521be/pngExample.png" width=512 height=512>

### Sensor Models
Volans uses sensor models to convert image coordinates to geodetic coordinates (lon/lat).  Volans uses three common sensor models:
1. 4-corner
2. Rational Polynomial Coefficient (RPC)
3. Replacement Sensor Model (RSM)

The 4-corner model is used only for orthorectified images.  The RPC and RSM models are used for raw images and require Digital Terrain Elevation Data (DTED) to convert between sample/line and lon/lat.

Volans uses a common trait called a GeoModel for all three of the sensor models.  The CvGdalReader creates the correct GeoModel based on the image metadata.  The image used here has an RPC model:
```
println(s"Sensor Model: ${gdalReader.geoModel}")
```
```
Sensor Model: GeoRpcModel(GeoRpcSerializable(com.ccri.volans.imagery.sensor_model.rpcCoeffs@2ca923bb,[Lcom.vividsolutions.jts.geom.Coordinate;@13df2a8c,None,),None)
```
The sensor model trait includes methods to convert coordinates between sample line and longitide and latitude:
```
val lonLat = gdalReader.geoModel.sampleLineToLonLat(sampleLine)
val sampleLineBack = gdalReader.geoModel.lonLatToSampleLine(lonLat)
println(s"Lon/Lat = $lonLat")
println(s"Sample/Line = $sampleLineBack")
```
```
Lon/Lat = (-58.52132826887419, -34.543130919256384, NaN)
Sample/Line = (302.40000003023306, 112.40000000278087, NaN)
```
Note that the JTS Coordinate data structure uses up to three coordinates.  The GeoModel uses only two, so the third is NaN.

The RPC and RSM sensor models must use DTED data for accurate results.  DTED level 2 is highly preferred while DTED level 0 is better than nothing but should be avoided.  The volans-imagery test module includes DTED level 2 coverage for this image, and here is an example of setting the DTED for the sensor model:
```
val cl = classOf[ImageryTest].getClassLoader
val dted2Path = cl.getResource("dted2").getPath
System.setProperty(ElevationModel.systemPropertyKey,dted2Path)
```
Note that the dted path must be set prior to creating the GeoModel.  A good practice is to set it after calling NativeLoader.init().  The DTED directory structure should look something like this:
```
/path/to/dted/w059/s35.dt2
```
Multiple dted paths can be set separated by a colon.  The lon/lat for the example above using dted 2 is:
```
Lon/Lat = (-58.52125456206289, -34.543142043978875, NaN)
```
Volans sensor models are integrated with JTS data structures.  Here is an example of calculating the bounding geodetic polygon for an image chip:
```
val sampLinePoly = GeomUtils.rectToPoly(rect)
val lonLatPoly = gdalReader.geoModel.project(sampLinePoly)
println(s"Sample/Line Polygon: $sampLinePoly")
println(s"lonLatPoly Polygon: $lonLatPoly")
```
```
Sample/Line Polygon: POLYGON ((512 512, 512 1024, 1024 1024, 1024 512, 512 512))
Lon/Lat Polygon: POLYGON ((-58.5221068272813 -34.54186276209387, -58.52213887351326 -34.54024863362048, -58.5242566899832 -34.54018622417615, -58.5242849558584 -34.541791231131725, -58.5221068272813 -34.54186276209387))
```
### Working with Raster Tiles
Volans includes a data structure called a CvLiteImage for working with rasters.  The CvLiteImage integrates the image pixel data with the sensor model and raster processing methods.  The class includes the following key fields:
* image: Map[Int, opencv_core.Mat] the image data for each channel
* geoModel: GeoModel a GeoModel specific to the tile

A CvLiteImage is created from the CvGdalReader in the following manner:
```
val rect = new opencv_core.Rect(512, 512, 512, 512)
val li = CvLiteImageConstruct(gdalReader, rect)
```
The GeoModel attached to the CvLiteImage has sample/line coordinates specific to the tile.  This example compares the lon/lat at the origin of the CvLiteImage to that location in the full image:
```
println(s"Lon/Lat at (512, 512) for CvGdalReader: ${gdalReader.geoModel.sampleLineToLonLat(new Coordinate(512, 512))}")
println(s"Lon/Lat at (0, 0) for CvLiteImage: ${li.geoModel.sampleLineToLonLat(new Coordinate(0, 0))}")

```
```
Lon/Lat at (512, 512) for CvGdalReader: (-58.5221068272813, -34.54186276209387, NaN)
Lon/Lat at (0, 0) for CvLiteImage: (-58.5221068272813, -34.54186276209387, NaN)

```
The CvLiteImage class has a number of methods available for raster processing.  The following example:
* Finds the min/max pixel values for each channel
* Orthorectifies the tile
* Sharpens the orthorectified tile using an unsharp mask
* Scales the sharpened tile
* Displays the scaled/sharpened/ortho'd tile in a window
* Writes the result to a GeoTiff

```
val liMinMax = li.minMax()
val liOrtho = li.ortho()
val liSharp = liOrtho.unsharp(unsharpSize = 7, unsharpSigma = 3.0, unsharpAlpha = 1.5, unsharpBeta = -0.5)
val liScaleSharp = liSharp.scaleMinMax(minMaxVals = Some(liMinMax))
liScaleSharp.show(scale = Some(1.5))
liScaleSharp.writeGeotTiff(new java.io.File(outPath, "liExample.tif").toString)
```

The GeoTiff can now be opened in QGIS:
<img src="/uploads/193ce9876375d8267c33a88c49a3e9d3/liExample.png">

It is a good idea to release Mats when done.  The CvLiteImage class includes a method to free all Mats associated with the class and the associated GeoModel:
```
li.free
liOrtho.free
liSharp.free
liScale.free
liScaleSharp.free
```
### Extending the CvLiteImage Class
CvLiteImage is designed to be readily updated with new raster processing methods.  CvLiteImage extends the following trait:
```
trait CvImageTrait {
  val image: Map[Int, Mat]
  val geoModel: GeoModel
  val fnOfImage: URI
  def free = {
    image map {case (k, v) => k -> v.release()}
    geoModel.free
  }
}
```
New raster processing methods should be a trait that extends the CvImageTrait.  As an example, consider the following trait that implements unsharping masking and returns a new sharpened CvLiteImage:
```
trait CvUnsharpTrait extends CvImageTrait {
  def unsharp(unsharpSize: Int = 7, unsharpSigma: Double = 3.0, unsharpAlpha: Double = 1.5, unsharpBeta: Double = -0.5) = {
    val result = image map { case (k, v) => k -> {
      val dst = new opencv_core.Mat()
      opencv_imgproc.GaussianBlur(v, dst, new opencv_core.Size(unsharpSize, unsharpSize), unsharpSigma)
      val sharp = new opencv_core.Mat()
      opencv_core.addWeighted(v, unsharpAlpha, dst, unsharpBeta, 0.0, sharp)
      dst.release()
      sharp
    }}
    CvLiteImage(result, geoModel, fnOfImage)
  }
}

```
To make the new method available on the CvLiteImage class, add it to the CvImageProcessingTrait in the following manner:
```
trait CvImageProcessingTrait extends CvImageTrait
  with CvShowTrait
  with CvWriteGeoTiffTrait
  with CvOrthoTrait
  with CvScaleMinMaxTrait
  with CvUnsharpTrait
```
The method "unsharp" is now available on the CvLiteImage class.

### Reading in Large Tiles
Satellite images can be large and reading the pixel values can take a long time.  This is particularly true of images that are JPEG2000 compressed.  Volans has a capability to parallelize reading an image tile.  The level of parallelization can be set when creating the CvGdalReader using the Option nCpu input:
```
val gdalReaderPar = CvGdalReader(imgFile, nCpu = Some(64))
```
Now evaluate the time to read the full 43,008 x 43,008 image (and then display it):
```
val ticLarge = System.currentTimeMillis()
val rectLarge = new opencv_core.Rect(0, 0, 43008, 43008)
val liLarge = CvLiteImageConstruct(gdalReaderPar, rectLarge)
println(s"Time to read full image (s) = ${(System.currentTimeMillis - ticLarge) / 1000d}")
liLarge.scaleMinMax().show(scale = Some(0.02))
```
```
Time to read full image (s) = 17.993
```
The single-threaded time to read the full image is over 5 minutes:
```
Time to read full image (s) = 304.443
```
### Working with Multi-Spectral Imagery
Volans works the same with multi-spectral imagery, but the band order should be specified.  Volans assumes that the band order is from shortest to longest wavelength, but the band order can be flipped if this is not the case.  The following image file has RGB band ordering.  This can be seen using the gdalinfo command line utility:
```
gdalinfo ~/data/po_344777_rgb_0000000.tif
```
```
...
Band 1 Block=24102x128 Type=Byte, ColorInterp=Red
Band 2 Block=24102x128 Type=Byte, ColorInterp=Green
Band 3 Block=24102x128 Type=Byte, ColorInterp=Blue
```
The following is an example of flipping the band order when reading the image data, creating a multi-spectral CvLightImage, and writing the tile as a PNG file.  
```
val msiFile = new java.io.File("/net/synds1/volume1/projects/dragonfish/images/po_344777_rgb_0000000.tif")
val gdalReaderMsi = CvGdalReader(msiFile, flipBands = Some(true))
val rectMsi = new opencv_core.Rect(12501, 12501, 512, 512)
val liMsi = CvLiteImageConstruct(gdalReaderMsi, rectMsi)
liMsi.imwrite("/path/to/output/msi.png")
```
<img src="/uploads/fdd7f8bc7ea412ed04c71ceb4c8f8eae/msi.png" width=512, height=512>
### Reading Images with Nitro
Volans offers integration with the Nitro NITF library as an alternative to GDAL.  The CvNitroReader can be used the same way as the CvGdalReader:
```
val nitroReader = CvNitroReader(imgFile, nCpu = Some(64), flipBands = Some(true))
val liNitro = CvLiteImageConstruct(nitroReader, rect)
```
## Command Line Tools
Volans includes a number of command line utilities.
### ImageChipper
ImageChipper chips an image to disk.  Example usage for chipping an orthorectified image with 78m 224x224 chips with 50% overlap.  Here we use 8 chipping workers and 8 writing workers:
```
java -cp volans-tools-0.1-SNAPSHOT.jar com.ccri.volans.tools.ImageChipper -imgFile /net/synds1/volume1/projects/dragonfish/images/po_344777_rgb_0000000.tif -outDirectory /path/to/output -numChipWorkers 8 -numWriteWorkers 8 -rectifier none -chipWidth 78.0 -chipSize 224 -chipOverlap 0.5 -flipBands yes
```
On the df01 machine, this example created 73,264 chips in approximately 28s.  

Now consider an image that has not previously been orthorectified and includes a sensor model.  The following creates chips rotated to the up-is-up orientation.  Since this is a larger image, use 64 total workers.  Trial and error shows that chipping takes longer than writing, so use 38 chipping workers and 24 writing workers.  Note that dted data is required to accurately rotate the chip to up-is-up.
```
java -cp volans-tools-0.1-SNAPSHOT.jar com.ccri.volans.tools.ImageChipper -imgFile /net/synds1/volume1/projects/dragonfish/multiview-stereo/01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF -outDirectory /path/to/output -numChipWorkers 38 -numWriteWorkers 24 -rectifier up -chipWidth 78.0 -chipSize 224 -chipOverlap 0.5 -dtedDir /net/synds1/volume1/projects/dragonfish/dted2 -chipCap 64
```
On the df01 machine, this example created 129,874 chips in approximately 52s.
### ImageVectorizer
ImageVectorizer breaks an image into overlapping chips, vectorizes each chip, and writes the feature vector to disk.  To run the code, make sure to use cuda version 9.1.  On the gpu05 machine:
```
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/targets/x86_64-linux/lib
```
An example use
```
java -cp volans-tools-0.1-SNAPSHOT.jar -Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS -Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.F2jLAPACK -Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.F2jARPACK -Xms10g -Xmx10g -Dorg.bytedeco.javacpp.maxbytes=10G -Dorg.bytedeco.javacpp.maxphysicalbytes=10G com.ccri.volans.tools.ImageVectorizer -imgFile /home/acochrane/data/imagery/01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF -outDirectory /home/acochrane/data/vectors -modelFile /home/acochrane/models/1.0.0-beta/resnet50_imagenet.zip -batchSize 32 -chipWidth 78.0 -chipSize 224 -channels 3 -chipOverlap 0.5 -numChipWorkers 8 -numWriters 6 -framework dl4j -dl4jAutoGc 5000 -gpuDevs 0 -rectifier up -preProc resnet -vecSize 2048 -numChipsPerGroup 4096 -tileSize 4096 -chipCap 32 -dtedDir /net/synds1/volume1/projects/dragonfish/dted2 -flipBands yes -useNitro no
```
On the gpu05 machine with a Titan X GPU, this example created and vectorized 129,874 chips in approximately 9.1 minutes.  

The image and model for this example are available at:
* /net/synds1/volume1/projects/dragonfish/multiview-stereo/01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF
* /net/synds1/volume1/projects/dragonfish/models/dl4j/1.0.0-beta/resnet50_imagenet.zip

The model is the ImageNet pretrained ResNet-50 model from the Dl4j model zoo.  Note: due to [this Dl4j issue](https://github.com/deeplearning4j/deeplearning4j/issues/5463), we are currently working on testing a ResNet-50 model imported to Dl4j from Keras. 

The input parameters are described here along with guidance to set the parameters.  Most of the parameters are suitable for running the ResNet-50 model on a Titan X GPU.
* imgFile: full path to the image file to vectorize
* modelFile: full path to the model to use to vectorize
* outDirectory: path to write the feature vectors
    * Each vector is written to disk as a serialized file
    * See below for reading the data into Scala/Java and Python
* batchSize: batch size to use for vectorizing
    * Default is 32, but larger batch sizes can be used for ResNet-50
    * Note that each feature vector is saved individually regardless of batch size
* chipWidth: physical size of the image chip
    * Default is 78.0m, but in general should be set for the desired application
    * Depends on:
        * Size of the object(s)
        * System resolution
    * Details of the implementation approach:
        * For orthorectified imagery, both dimensions of the chip are this size
        * For up-is-up imagery, the horizontal dimension is this size.  The vertical dimension is determined by the collection geometry.
* chipSize: the dimensions of the chip 
    * Default is 224
    * Chip is assumed to be square, will update if necessary
* channels: the number of channels for the model
    * Default is 3
    * For panchromatic data, the single band is replicated accross the channels
* chipOverlap: chip overlap
    * Default is 0.5 (50% overalap)
* numChipWorkers: number of chipping workers
    * Default is 1
    * Set high enough to keep the GPU(s) busy
    * Assuming that the image chipping process is faster than forward inference, 4-8 workers is more than sufficient
* numCpuWorkers: if using CPU(s) for forward inference, number of workers to use
    * Default is 1
    * No effect if using GPU(s)
    * Dl4j parallelizes under the hood, so experiment to find the right value (probably 4-5)
* numWriters: number of feature writers
    * Default is 1
    * During processing, keep an eye on "Number of detections in the queue" in the log.  If it grows unbounded, more feature writers are required.
    * Setting this to 6 was more than sufficient in the above example and will likely work for most cases
* framework: deep learning framework to use for forward inference
    * Default is dl4j, currently the only available Volans framework
    * Goal is to add others (Tensorflow, Caffe, PyTorch, ...) as Java bindings become available
* dl4jAutoGc: Dl4j/Nd4j garbage collection interval (ms)
    * Default is 50000
    * In the above example, set to 5000 to minimize memory usage at the slight expense of speed
* gpuDevs: GPU devices to use for processing (comma-delimited string)
    * Default is "0"
* rectifier: chip rectifier (Options are "ortho", "up", or "none")
    * Default is "none"
    * "ortho" and "up" require a sensor model (RPC or RSM)
    * For imagery that is already orthorectified, use "none"
    * For imagery with a sensor model, recommend using "up" (up-is-up).  
        * Eliminates rotational ambiguity
        * No need for rotational training chip augmentation
* preProc: pre-processor (only option is "resnet")
    * Default is "resnet"
        * Stretch each chip to 8-bits (0-255)
        * For each band, subtract the mean channel value from the ImageNet dataset
    * Setting this to anything else will result in no preprocessing
* vecSize: feature vector size
    * Default is 2048 for ResNet-50
    * TODO: determine from the model
* numChipsPerGroup: the maximum size of a group of chips
    * Default is 4096
    * A chip group is the number of chips sent to the chip queue and the processed by a GPU/CPU worker
    * Considerations are memory (don't set too high) and keeping the worker busy (don't set too low)
    * Default is sufficient for most applications
* tileSize: size of an image tile for a chipping worker
    * Default is 4096
    * Each chipping worker creates image chips from this size tile
    * Default is sufficient for most applications
* chipCap: chip queue capacity
    * Default is 32
    * Chipping is typically faster than forward inference
    * chip queue is capped to prevent excessive memory use
    * Default is sufficient for most applications
* dtedDir: path to DTED data
    * Default is null
    * Sensor models require DTED data for accurate geolocation
    * Strongly recommend level 2 for "ortho" and "up" rectifiers
* flipBands: flip bands for multispetral imagery
    * Default is "yes"
    * Volans expects channels to run shortest to longest wavelength (e.g. BGR)
    * Flip if this is not the case
    * No effect for panchromatic imagery
* useNitro: use the Nitro NITF library rather than GDAL
    * Default is "no"
    * For most applications this is sufficent since GDAL can read NITFs
    * Dragonspell users should familiarize themselves with Nitro
        * Handles any size image, GDAL fails for very large images
        * Nitro has better TRE support

#### Reading the Feature Vectors in Scala
```
val vectorSize = 2048
val ios = new FileInputStream(vectorFile.toString)
val dis = new java.io.DataInputStream(ios)
val vector = (0 until vectorSize).map(idx => dis.readFloat())

```
#### Reading the Feature Vectors in Python
```
import numpy as np
filename = "/path/to/file/filename.bin"
with open(filename, 'rb') as f:
    data = np.fromFile(f, '>f4')
```
## Upcoming Capabilities
Volans is consolidating capabilities from Dragonfish as well as other sources.  Upcoming capabilities will focus on:
1. Application of Convolutional Neural Networks on multiple-GPU machines
2. Application of CNNs at scale within AWS
3. Creating, Training, Testing, and Deploying CNNs

This document will be updated as the capabilities are added.  
