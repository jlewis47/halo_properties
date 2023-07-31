import numpy as np
import os

def read_avgs(path, name=None):        

    if name==None:name="averages.txt"

    try:                                                                                             
        cols = ['aexp','nH','x','x*nH','T','J','S','nb_stars']                                       
        datas = np.genfromtxt(os.path.join(path,name),skip_header=1)                       
        format_datas=np.reshape(np.ravel(datas),(int(len(datas)/3),9))                               
        return({key:col for col,key in zip(np.transpose(format_datas[:,[0,2,3,4,5,6,7,8]]),cols)})   
    except:                                                                                          
        cols = ['aexp','nH','x','x*nH','T','J','Z','d','S','nb_stars']                               
                                                                                                     
        with open(os.path.join(path,name), 'r') as src:                                    
            data_lines=[]                                                                            
            tmp_line=[]                                                                              
            for line in src:                                                                         
                if '#' in line:continue                                                              
                tmp=line.strip('\n').split(' ')                                                      
                tmp=[entry for entry in tmp if entry!='']                                            
                if len(tmp_line)<11:                                                                 
                    tmp_line.extend(tmp)                                                             
                else:                                                                                
                    data_lines.append(np.float64(tmp_line))                                          
                    tmp_line=tmp                                                                     
                                                                                                     
        format_datas=np.asarray(data_lines)                                                          
        return({key:col for col,key in zip(np.transpose(format_datas[:,[0,2,3,4,5,6,7,8,9,10]]),cols\
)})    