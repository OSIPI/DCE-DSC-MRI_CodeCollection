import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
import datetime, time
import os
import progressbar
import numpy as np
from .MRImage import *


def loadDicomDir(path, each=None, load=None):
    """
    Function that loads all dicom files ina directory.

    Args:
            path:
                    path to the directory of the dicom files
            each:
                    if each is not none, only every each-th file will
                    be loaded
            load:
                    list of file indeces to be loaded

    returns:
            MRRawImage object
    """

    # find available dicom files
    dicomFilesList = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                dicomFilesList.append(os.path.join(dirName, filename))

    dicomFilesList = np.array(dicomFilesList)

    # checking and handling each and load arguments
    if (each is not None) and (load is not None):
        printWarning(
            "I am confused. Cannot use both each and load at the same time. Using only load."
        )
        each = None

    if type(load) is list:
        dicomFilesList = dicomFilesList[load]
    if each is not None:
        dicomFilesList = dicomFilesList[0::each]

    # The dicom objects are not necessarily ordered correcty
    # Place all dcm objects in a list for further processing
    dicomObjectList = []
    pbar = ProgressBar("Loading dicom...", len(dicomFilesList))
    for filename in dicomFilesList:
        dicomObjectList.append(dicom.read_file(filename))
        pbar.update()
    pbar.finish()

    # the next bit is to extract all the relevant information from
    # the dicom files
    voxelArray = []
    time = []
    sliceLocations = []

    # for use if the dicom objects to not contain time info
    currTime = 0
    dt = 3400  # ms
    if not hasattr(dicomObjectList[0], "TriggerTime"):
        printWarning("No time info found in Dicom. Using 3.4 time resolution.")

    for dcm in dicomObjectList:
        try:
            sliceLoc = float(dcm.SliceLocation)
        except AttributeError:
            sliceLog = 0
        if not sliceLoc in sliceLocations:
            sliceLocations.append(sliceLoc)
            voxelArray.append([])
            time.append([])

        # find the index of the current slice location
        idx = np.where(np.array(sliceLocations) == sliceLoc)[0][0]

        # append the slice data to the respective place
        voxelArray[idx].append(dcm.pixel_array)

        try:
            time[idx].append(float(dcm.TriggerTime))
        except AttributeError:
            curr_time += dt
            time[idx].append(curr_time)

    # make the lists numpy arrays for faster processing
    voxelArray = np.array(voxelArray)
    time = np.array(time)
    sliceLocations = np.array(sliceLocations)

    # sort the slices in order of slice location
    argsort = np.argsort(sliceLocations)
    voxelArray = voxelArray[argsort]
    sliceLocations = sliceLocations[argsort]

    # Order the time points of each slice in ascending order
    time = time[
        0
    ]  # assuming the time array is the same for all slices (as they should be)
    time = time - time[0]
    argsort = np.argsort(time)
    time = time[argsort]
    for i in range(len(voxelArray)):
        voxelArray[i] = voxelArray[i][argsort]

    time = time / 1000.0  # time in seconds

    MRImageArguments = {
        "voxelArray": voxelArray,
        "time": time,
        "sliceLocations": sliceLocations,
        "FA": dicomObjectList[0].FlipAngle * np.pi / 180.0,  # in radians
        "TR": dicomObjectList[0].RepetitionTime / 1000.0,  # in seconds
    }

    return MRRawImage(**MRImageArguments)


def writeDicom(pixel_array, filename):
    """
    Source: https://codedump.io/share/qCDN4fOKcTAS/1/create-pydicom-file-from-numpy-array

    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "Secondary Capture Image Storage"
    file_meta.MediaStorageSOPInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780"
    )
    file_meta.ImplementationClassUID = "1.3.6.1.4.1.9590.100.1.0.100.4.0"
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Modality = "WSD"
    ds.ContentDate = str(datetime.date.today()).replace("-", "")
    ds.ContentTime = str(time.time())  # milliseconds since the epoch
    ds.StudyInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093"
    )
    ds.SeriesInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649"
    )
    ds.SOPInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780"
    )
    ds.SOPClassUID = "Secondary Capture Image Storage"
    ds.SecondaryCaptureDeviceManufctur = "Python 2.7.3"

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16

    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)
    return


class DisplayFunctions:
    def __init__(self):
        self.timePlotParams = {"xlabel": "time (s)", "ylabel": "Signal Intensity"}

    def imshow(self, data_array, **kwargs):
        """
        plots and shows data in a 5D array, data_array.
        Should contain images arranged by slices, then time points

        args:
                data_array[slice][time_point][ypos, xpos] (or data_array[slice, time_point, ypos, xpos])
                show (bool): should the figure be shown immidiately?
                ax (matplotlib.pyplot.Axes): axes object to draw the figure on
                time (array like): list of time points
        """
        defaults = {
            "showfig": False,
            "ax": None,
            "cmap": "gray",
            "vmin": None,
            "vmax": None,
            "title": "",
            "xlabel": "",
            "ylabel": "",
        }
        print(kwargs)
        for key in defaults:
            kwargs.setdefault(key, defaults[key])

        showfig = kwargs["showfig"]
        ax = kwargs["ax"]
        cmap = kwargs["cmap"]
        vmin = kwargs["vmin"]
        vmax = kwargs["vmax"]
        title = kwargs["title"]
        xlabel = kwargs["xlabel"]
        ylabel = kwargs["ylabel"]

        if ax is None:
            # initiate figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        self.mainfigure = ax.figure
        self.mainax = ax

        ax.set_xticks([])
        ax.set_yticks([])
        plt.subplots_adjust(left=0.25, bottom=0.25)
        ax.data_array = data_array
        ax.currentSliceIdx = 0

        # initiate the image
        image = ax.imshow(
            data_array[0, 0],
            cmap=plt.get_cmap(cmap),
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        # add sliders to move through slices and timepoints
        sliceAx = sliceAx = inset_axes(
            ax,
            width="100%",  # width = 10% of parent_bbox width
            height="5%",  # height : 50%
            loc=3,
            bbox_to_anchor=(0.0, -0.1, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        sliceSlider = Slider(
            sliceAx, "Slice", 0, len(data_array) - 1, valinit=0, valfmt="%i"
        )

        timeAx = inset_axes(
            ax,
            width="100%",  # width = 10% of parent_bbox width
            height="5%",  # height : 50%
            loc=3,
            bbox_to_anchor=(0.0, -0.17, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        timeSlider = Slider(
            timeAx, "Time", 0, len(data_array[0]) - 1, valinit=0, valfmt="%i"
        )
        if len(data_array) - 1 == 0:
            sliceAx.set_visible(False)
        if len(data_array[0]) - 1 == 0:
            timeAx.set_visible(False)

        # global variables to store the current slice and time index
        currentSliceIdx = 0
        currentTimeidx = 0

        def update(val):
            global currentTimeidx
            global currentSliceIdx
            valTime = int(timeSlider.val)
            valSlice = int(sliceSlider.val)

            if valTime >= len(data_array[0]):
                currentTimeidx = len(data_array[0] - 1)
            else:
                currentTimeidx = valTime

            if valSlice >= len(data_array):
                currentSliceIdx = len(data_array) - 1
            else:
                currentSliceIdx = valSlice

            ax.currentSliceIdx = currentSliceIdx

            image.set_data(data_array[currentSliceIdx, currentTimeidx])
            ax.draw_artist(image)

        sliceSlider.on_changed(update)
        timeSlider.on_changed(update)

        if showfig:
            plt.show()

        return ax


def printWarning(string):
    print("WARNING!: {}".format(string))


def macOSnotify(title, content):
    os.system(
        """osascript -e 'display notification "{}" with title "{}"' """.format(
            content, title
        )
    )


class ProgressBar:
    def __init__(self, descriptor, maxval, widgets=None, show=True):
        self.show = show
        self.maxval = maxval
        self.widgets = widgets
        self.pbar_counter = 0
        if widgets is None and show:
            self.widgets = [
                " {:25.25}".format(descriptor),
                " ",
                progressbar.Bar(),
                " ",
                progressbar.Percentage(),
                " ",
                progressbar.ETA(),
            ]
        if show:
            self.pbar = progressbar.ProgressBar(
                maxval=self.maxval, widgets=self.widgets
            ).start()

    def update(self, i=0, descriptor=None):
        if self.show:
            self.pbar.update(self.pbar_counter + i)
            if descriptor is not None:
                self.widgets[0] = " {:25.25}".format(descriptor)
            self.pbar_counter += 1 + i

    def finish(self):
        if self.show:
            for i in range(len(self.widgets)):
                if type(self.widgets[i]) == type(progressbar.Percentage()):
                    self.widgets.pop(i)
                    self.widgets.pop(i)
                    break
            self.pbar.finish()


def saveAsTableText(value, shape, fname):
    value = np.array(value)
    if shape and len(value.shape) < 2:
        a = np.zeros(shape)
        a += value
        value = a
    np.savetxt(fname, value, "%.6g")
