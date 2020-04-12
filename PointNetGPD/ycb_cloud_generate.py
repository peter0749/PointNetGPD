import os
import numpy as np
import numba as nb
import h5py as h5
from scipy.misc import imread
from struct import pack, unpack
import math
import multiprocessing as mp
import glob


# extract pointcloud from rgb-d, then convert it into obj coordinate system(tsdf/poisson reconstructed object)
@nb.jit(nopython=True)
def im2col_nb(im, psize):
    n_channels, rows, cols = im.shape[0], im.shape[1], im.shape[2]

    im_pad = np.zeros((n_channels,
                       int(math.ceil(1.0 * rows / psize) * psize),
                       int(math.ceil(1.0 * cols / psize) * psize)), dtype=np.float32)
    im_pad[:, 0:rows, 0:cols] = im

    final = np.zeros((im_pad.shape[1], im_pad.shape[2], n_channels,
                      psize, psize), dtype=np.float32)
    for c in range(n_channels):
        for x in range(psize):
            for y in range(psize):
                im_shift = np.vstack(
                    (im_pad[c, x:], im_pad[c, :x]))
                im_shift = np.column_stack(
                    (im_shift[:, y:], im_shift[:, :y]))
                shape = (int(im_pad.shape[1] / psize), psize,
                         int(im_pad.shape[2] / psize), psize)
                final[x::psize, y::psize, c] = im_shift.reshape(shape[0], shape[1]*shape[2], shape[3]).T.reshape(shape[3], shape[1], shape[2], shape[0]).T

    return final[0:rows - psize + 1, 0:cols - psize + 1]

def filterDiscontinuities(depthMap):
    filt_size = 7
    thresh = 1000

    # Compute discontinuities
    offset = int((filt_size - 1) / 2)
    patches = im2col_nb(depthMap[np.newaxis], filt_size)[:,:,0]
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids),
                         np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros((480, 640), dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depthMap * (1 - final_mark)


@nb.jit(nopython=True)
def registerDepthMap_nb(unregisteredDepthMap,
                     rgbImage,
                     depthK,
                     rgbK,
                     H_RGBFromDepth):


    unregisteredHeight = unregisteredDepthMap.shape[0]
    unregisteredWidth = unregisteredDepthMap.shape[1]

    registeredHeight = rgbImage.shape[0]
    registeredWidth = rgbImage.shape[1]

    registeredDepthMap = np.zeros((registeredHeight, registeredWidth))

    xyzDepth = np.ones((1,4), dtype=np.float32)

    invDepthFx = 1.0 / depthK[0,0]
    invDepthFy = 1.0 / depthK[1,1]
    depthCx = depthK[0,2]
    depthCy = depthK[1,2]

    rgbFx = rgbK[0,0]
    rgbFy = rgbK[1,1]
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]

    undistorted = np.empty(2)
    for v in range(unregisteredHeight):
      for u in range(unregisteredWidth):

            depth = unregisteredDepthMap[v,u]
            if depth == 0:
                continue

            xyzDepth[0,0] = ((u - depthCx) * depth) * invDepthFx
            xyzDepth[0,1] = ((v - depthCy) * depth) * invDepthFy
            xyzDepth[0,2] = depth

            xyzRGB = np.sum(H_RGBFromDepth*xyzDepth, axis=1) # shape: (3,)

            invRGB_Z  = 1.0 / xyzRGB[2]
            undistorted[0] = (rgbFx * xyzRGB[0]) * invRGB_Z + rgbCx
            undistorted[1] = (rgbFy * xyzRGB[1]) * invRGB_Z + rgbCy

            uRGB = int(undistorted[0] + 0.5)
            vRGB = int(undistorted[1] + 0.5)

            if (uRGB < 0 or uRGB >= registeredWidth) or (vRGB < 0 or vRGB >= registeredHeight):
                continue

            registeredDepth = xyzRGB[2]
            if registeredDepth > registeredDepthMap[vRGB,uRGB]:
                registeredDepthMap[vRGB,uRGB] = registeredDepth

    return registeredDepthMap


@nb.jit(nopython=True)
def registeredDepthMapToPointCloud_nb(depthMap, rgbImage, rgbK, refFromRGB, objFromref):
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]
    invRGBFx = 1.0/rgbK[0,0]
    invRGBFy = 1.0/rgbK[1,1]

    height = depthMap.shape[0]
    width = depthMap.shape[1]

    cloud = np.empty((1, height*width, 6), dtype=np.float32)

    goodPointsCount = 0
    for v in range(height):
        for u in range(width):

            depth = depthMap[v,u]

            row = 0
            col = goodPointsCount

            if depth <= 0:
                continue

            x = (u - rgbCx) * depth * invRGBFx
            y = (v - rgbCy) * depth * invRGBFy
            z = depth

            xyz1 = np.array([x,y,z,1], dtype=np.float32).reshape(1, 4) # shape: (1, 4): xyz1

            assert len(refFromRGB.shape)==2
            xyz = np.sum(refFromRGB*xyz1, axis=1) # shape: (3, )
            xyz1[0,:3] = xyz[:3] # update xyz1

            assert len(objFromref.shape)==2
            # obj from ref
            cloud[row,col,:3] = np.sum(objFromref[:3,:4]*xyz1, axis=1)

            cloud[row,col,3] = rgbImage[v,u,0]
            cloud[row,col,4] = rgbImage[v,u,1]
            cloud[row,col,5] = rgbImage[v,u,2]
            goodPointsCount += 1

    cloud = cloud[:,:goodPointsCount,:]

    return cloud


def writePLY(filename, cloud, faces=[]):
    if len(cloud.shape) != 3:
        print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(cloud.shape))
        return

    color = True if cloud.shape[2] == 6 else False
    num_points = cloud.shape[0]*cloud.shape[1]

    header_lines = [
        'ply',
        'format ascii 1.0',
        'element vertex %d' % num_points,
        'property float x',
        'property float y',
        'property float z',
        ]
    if color:
        header_lines.extend([
        'property uchar diffuse_red',
        'property uchar diffuse_green',
        'property uchar diffuse_blue',
        ])
    if faces != None:
        header_lines.extend([
        'element face %d' % len(faces),
        'property list uchar int vertex_indices'
        ])

    header_lines.extend([
      'end_header',
      ])

    f = open(filename, 'w+')
    f.write('\n'.join(header_lines))
    f.write('\n')

    lines = []
    for i in range(cloud.shape[0]):
        for j in range(cloud.shape[1]):
            if color:
                lines.append('%s %s %s %d %d %d' % tuple(cloud[i, j, :].tolist()))
            else:
                lines.append('%s %s %s' % tuple(cloud[i, j, :].tolist()))

    for face in faces:
        lines.append(('%d' + ' %d'*len(face)) % tuple([len(face)] + list(face)))

    f.write('\n'.join(lines) + '\n')
    f.close()

def writePCD(filename, pointCloud, ascii=True):
    if len(pointCloud.shape) != 3:
      print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(pointCloud.shape))
      return
    with open(filename, 'w') as f:
        height = pointCloud.shape[0]
        width = pointCloud.shape[1]
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        if pointCloud.shape[2] == 3:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        else:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH %d\n" % width)
        f.write("HEIGHT %d\n" % height)
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS %d\n" % (height * width))
        if ascii:
          f.write("DATA ascii\n")
          for row in range(height):
            for col in range(width):
                if pointCloud.shape[2] == 3:
                    f.write("%f %f %f\n" % tuple(pointCloud[row, col, :]))
                else:
                    f.write("%f %f %f" % tuple(pointCloud[row, col, :3]))
                    r = int(pointCloud[row, col, 3])
                    g = int(pointCloud[row, col, 4])
                    b = int(pointCloud[row, col, 5])
                    rgb_int = (r << 16) | (g << 8) | b
                    packed = pack('i', rgb_int)
                    rgb = unpack('f', packed)[0]
                    f.write(" %.12e\n" % rgb)
        else:
          f.write("DATA binary\n")
          if pointCloud.shape[2] == 6:
              # These are written as bgr because rgb is interpreted as a single
              # little-endian float.
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('b', np.uint8),
                             ('g', np.uint8),
                             ('r', np.uint8),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z', 'r', 'g', 'b']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)
          else:
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)

def getRGBFromDepthTransform(calibration, camera, referenceCamera):
    irKey = "H_{0}_ir_from_{1}".format(camera, referenceCamera)
    rgbKey = "H_{0}_from_{1}".format(camera, referenceCamera)

    rgbFromRef = calibration[rgbKey][:]
    irFromRef = calibration[irKey][:]

    return np.dot(rgbFromRef, np.linalg.inv(irFromRef)), np.linalg.inv(rgbFromRef)


def generate(path):
    path = path.split('/')
    # Parameters
    ycb_data_folder = path[0]+'/'+path[1]            # Folder that contains the ycb data.
    target_object = path[2]    # Full name of the target object.
    viewpoint_camera = path[3].split('_')[0]                # Camera which the viewpoint will be generated.
    viewpoint_angle = path[3].split('_')[1].split('.')[0]                # Relative angle of the object w.r.t the camera (angle of the turntable).

    referenceCamera = "NP5" # can only be NP5

    ply_fname = os.path.join(ycb_data_folder, target_object, "clouds", "pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+".ply")
    pcd_fname = os.path.join(ycb_data_folder, target_object, "clouds", "pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+".pcd")
    npy_fname = os.path.join(ycb_data_folder, target_object, "clouds", "pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+".npy")

    '''
    if os.path.exists(ply_fname) and os.path.exists(pcd_fname):
        print(ycb_data_folder, target_object, viewpoint_camera, viewpoint_angle, 'pass')
        return
    else:
    '''
    print(ycb_data_folder, target_object, viewpoint_camera, viewpoint_angle)

    #if True:
    try:
        if not os.path.exists(os.path.join(ycb_data_folder, target_object, "clouds")):
            os.makedirs(os.path.join(ycb_data_folder, target_object, "clouds"))

        basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
        depthFilename = os.path.join(ycb_data_folder, target_object, basename + ".h5")
        rgbFilename = os.path.join(ycb_data_folder, target_object, basename + ".jpg")
        pbmFilename = os.path.join(ycb_data_folder, target_object, 'masks', basename + "_mask.pbm")

        calibrationFilename = os.path.join(ycb_data_folder, target_object, "calibration.h5")
        objFromrefFilename = os.path.join(ycb_data_folder, target_object, 'poses', '{0}_{1}_pose.h5'.format(referenceCamera, viewpoint_angle))
        calibration = h5.File(calibrationFilename, 'r')
        objFromref = h5.File(objFromrefFilename, 'r')['H_table_from_reference_camera'][:]

        if not os.path.isfile(rgbFilename):
            print("The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
            exit(1)
        rgbImage = imread(rgbFilename)
        pbmImage = imread(pbmFilename)
        depthK = calibration["{0}_depth_K".format(viewpoint_camera)][:] # use depth instead of ir
        rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]
        depthScale = np.array(calibration["{0}_ir_depth_scale".format(viewpoint_camera)]) * .0001 # 100um to meters
        H_RGBFromDepth, refFromRGB = getRGBFromDepthTransform(calibration, viewpoint_camera, referenceCamera)

        unregisteredDepthMap = h5.File(depthFilename, 'r')["depth"][:]
        unregisteredDepthMap = filterDiscontinuities(unregisteredDepthMap) * depthScale

        registeredDepthMap = registerDepthMap_nb(unregisteredDepthMap,
                                              rgbImage,
                                              depthK,
                                              rgbK,
                                              H_RGBFromDepth)
        # apply mask
        registeredDepthMap[pbmImage == 255] = 0

        pointCloud = registeredDepthMapToPointCloud_nb(registeredDepthMap, rgbImage, rgbK, refFromRGB, objFromref)

        writePLY(ply_fname, pointCloud)
        writePCD(pcd_fname, pointCloud)
        np.save(npy_fname, pointCloud[: ,:, :3].reshape(-1, 3))
    except Exception as e:
        f = open('exception.txt', 'a')
        f.write('/'.join(path) + ': ' + str(e) + '\n')
        f.close()
        print(ycb_data_folder, target_object, viewpoint_camera, viewpoint_angle, 'failed', str(e))
        if os.path.exists(ply_fname):
            os.unlink(ply_fname)
        if os.path.exists(pcd_fname):
            os.unlink(pcd_fname)
        if os.path.exists(npy_fname):
            os.unlink(npy_fname)
        return


if __name__ == "__main__":
    fl = np.array(glob.glob('data/ycb_rgbd/0*/*.jpg'))
    np.random.shuffle(fl)
    cores = 20 # mp.cpu_count()
    pool = mp.Pool(processes=cores)
    pool.map(generate, fl)
    #for v in fl:
    #    generate(v)
