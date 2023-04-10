import sys
import getopt
import traceback
import numpy as np
import FileManager
import Visualization
import LungSegmentation
import BronchusSegmentation
import VascularSegmentation


INPUT_PATH = './data/case_1/DICOM'
OUTPUT_PATH = './output/case_1'

opts, args = getopt.gnu_getopt(sys.argv[1:], '', ['input=', 'output='])
for opt in opts :
    if opt[0] == '--input' :
        INPUT_PATH = opt[1]
    elif opt[0] == '--output' :
        OUTPUT_PATH = opt[1]

RAW_MASK_PATH = OUTPUT_PATH + '/_raw_mask'
LUNG_PATH = OUTPUT_PATH + '/_lung_mask'
BRONCHUS_PATH = OUTPUT_PATH + '/_bronchus_tree'
VASCULAR_PATH = OUTPUT_PATH + '/_vascular_tree'

IMAGE_SCALE = (512, 512)

try :
    #data = FileManager.ReadDicomDir(INPUT_PATH, IMAGE_SCALE)

    #raw_mask, lung_mask = LungSegmentation.Execute(data)
    #FileManager.Write3DArrayAsImages(RAW_MASK_PATH, raw_mask)
    #FileManager.Write3DArrayAsImages(LUNG_PATH, lung_mask)
    #raw_mask = FileManager.ReadPNGDir(RAW_MASK_PATH, IMAGE_SCALE)
    lung_mask = FileManager.ReadPNGDir(LUNG_PATH, IMAGE_SCALE)
    Visualization.Draw3DContours(lung_mask)

    #bronchus_tree = BronchusSegmentation.Execute(data, lung_mask)
    #FileManager.Write3DArrayAsImages(BRONCHUS_PATH, bronchus_tree)
    bronchus_tree = FileManager.ReadPNGDir(BRONCHUS_PATH, IMAGE_SCALE)
    Visualization.Draw3DContours(bronchus_tree, color = (1, 1, 0))
    
    #vascular_tree = VascularSegmentation.Execute(data, raw_mask, lung_mask, bronchus_tree, VASCULAR_PATH)
    #FileManager.Write3DArrayAsImages(VASCULAR_PATH, vascular_tree)
    vascular_tree = FileManager.ReadPNGDir(VASCULAR_PATH, IMAGE_SCALE)
    Visualization.Draw3DContours(vascular_tree, color = (1, 0, 0))

    node = np.load(OUTPUT_PATH + "/predict.npy")
    Visualization.Draw3DContours(node, color = (0, 1, 0))

    labels = np.zeros(node.shape, dtype = np.uint8)
    labels[vascular_tree != 0] = 1
    labels[node != 0] = 2
    FileManager.Write3DArrayAsNIFTI(OUTPUT_PATH + "/label.nii", labels)
    
    Visualization.Show()

except Exception as e :
    print(f'Case Failed: {INPUT_PATH}')
    print(e.args)
    print(traceback.format_exc())

