
import tensorflow as tf
import numpy as np

# Functions taken from original IM-NET implementation: https://github.com/czq142857/IM-NET-pytorch/

def z2voxel(z, autoencoder):
    cell_grid_size = 4
    frame_grid_size = 64
    real_size = cell_grid_size*frame_grid_size
    test_size = 32
    test_point_batch_size = test_size*test_size*test_size

    dimc = cell_grid_size
    dimf = frame_grid_size
    cell_x = np.zeros([dimc,dimc,dimc],np.int32)
    cell_y = np.zeros([dimc,dimc,dimc],np.int32)
    cell_z = np.zeros([dimc,dimc,dimc],np.int32)
    cell_coords = np.zeros([dimf,dimf,dimf,dimc,dimc,dimc,3],np.float32)
    frame_coords = np.zeros([dimf,dimf,dimf,3],np.float32)
    frame_x = np.zeros([dimf,dimf,dimf],np.int32)
    frame_y = np.zeros([dimf,dimf,dimf],np.int32)
    frame_z = np.zeros([dimf,dimf,dimf],np.int32)
    for i in range(dimc):
        for j in range(dimc):
            for k in range(dimc):
                cell_x[i,j,k] = i
                cell_y[i,j,k] = j
                cell_z[i,j,k] = k
    for i in range(dimf):
        for j in range(dimf):
            for k in range(dimf):
                cell_coords[i,j,k,:,:,:,0] = cell_x+i*dimc
                cell_coords[i,j,k,:,:,:,1] = cell_y+j*dimc
                cell_coords[i,j,k,:,:,:,2] = cell_z+k*dimc
                frame_coords[i,j,k,0] = i
                frame_coords[i,j,k,1] = j
                frame_coords[i,j,k,2] = k
                frame_x[i,j,k] = i
                frame_y[i,j,k] = j
                frame_z[i,j,k] = k
    cell_coords = (cell_coords.astype(np.float32)+0.5)/real_size-0.5
    cell_coords = np.reshape(cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
    cell_x = np.reshape(cell_x,[dimc*dimc*dimc])
    cell_y = np.reshape(cell_y,[dimc*dimc*dimc])
    cell_z = np.reshape(cell_z,[dimc*dimc*dimc])
    frame_x = np.reshape(frame_x,[dimf*dimf*dimf])
    frame_y = np.reshape(frame_y,[dimf*dimf*dimf])
    frame_z = np.reshape(frame_z,[dimf*dimf*dimf])
    frame_coords = (frame_coords.astype(np.float32)+0.5)/dimf-0.5
    frame_coords = np.reshape(frame_coords,[dimf*dimf*dimf,3])
    
    sampling_threshold = 0.5

    model_float = np.zeros([real_size+2,real_size+2,real_size+2],np.float32)
    dimc = cell_grid_size
    dimf = frame_grid_size
    
    frame_flag = np.zeros([dimf+2,dimf+2,dimf+2],np.uint8)
    queue = []
    
    frame_batch_num = int(dimf**3/test_point_batch_size)
    assert frame_batch_num>0
    
    #get frame grid values
    for i in range(frame_batch_num):
        point_coord = frame_coords[i*test_point_batch_size:(i+1)*test_point_batch_size]
        point_coord = np.expand_dims(point_coord, axis=0)
        _, model_out_ = autoencoder(None, z, point_coord, is_training=False)
        model_out = model_out_.detach().cpu().numpy()[0]
        x_coords = frame_x[i*test_point_batch_size:(i+1)*test_point_batch_size]
        y_coords = frame_y[i*test_point_batch_size:(i+1)*test_point_batch_size]
        z_coords = frame_z[i*test_point_batch_size:(i+1)*test_point_batch_size]
        frame_flag[x_coords+1,y_coords+1,z_coords+1] = np.reshape((model_out>sampling_threshold).astype(np.uint8), [test_point_batch_size])
    
    #get queue and fill up ones
    for i in range(1,dimf+1):
        for j in range(1,dimf+1):
            for k in range(1,dimf+1):
                maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
                minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
                if maxv!=minv:
                    queue.append((i,j,k))
                elif maxv==1:
                    x_coords = cell_x+(i-1)*dimc
                    y_coords = cell_y+(j-1)*dimc
                    z_coords = cell_z+(k-1)*dimc
                    model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0
    
    print("running queue:",len(queue))
    cell_batch_size = dimc**3
    cell_batch_num = int(test_point_batch_size/cell_batch_size)
    assert cell_batch_num>0
    #run queue
    while len(queue)>0:
        batch_num = min(len(queue),cell_batch_num)
        point_list = []
        cell_coords = []
        for i in range(batch_num):
            point = queue.pop(0)
            point_list.append(point)
            cell_coords.append(cell_coords[point[0]-1,point[1]-1,point[2]-1])
        cell_coords = np.concatenate(cell_coords, axis=0)
        cell_coords = np.expand_dims(cell_coords, axis=0)
        _, model_out_batch = autoencoder(None, z, cell_coords, is_training=False)
        for i in range(batch_num):
            point = point_list[i]
            model_out = model_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,0]
            x_coords = cell_x+(point[0]-1)*dimc
            y_coords = cell_y+(point[1]-1)*dimc
            z_coords = cell_z+(point[2]-1)*dimc
            model_float[x_coords+1,y_coords+1,z_coords+1] = model_out
            
            if np.max(model_out)>sampling_threshold:
                for i in range(-1,2):
                    pi = point[0]+i
                    if pi<=0 or pi>dimf: continue
                    for j in range(-1,2):
                        pj = point[1]+j
                        if pj<=0 or pj>dimf: continue
                        for k in range(-1,2):
                            pk = point[2]+k
                            if pk<=0 or pk>dimf: continue
                            if (frame_flag[pi,pj,pk] == 0):
                                frame_flag[pi,pj,pk] = 1
                                queue.append((pi,pj,pk))
    return model_float

def write_ply_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()