# -*- coding: utf-8 -*-

#python 自动获取（打印）代码中的变量的名字字串
import glob, os


def objname(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def print_more(v_name,v):
    """ print 变量的名字及type、及其值 """
    print("{0}, type = {1}, value = ".format(v_name,type(v)))
    print(v)
    try:
        print("shape = {0}".format(v.shape))
    except:
        try:
            print("len = {0}".format(len(v)))
        except:
            pass


def print_mnist_dataset(mnist):
    print_more("mnist.train.labels",mnist.train.labels)
    print_more("mnist.validation.labels",mnist.validation.labels)
    print_more("mnist.test.labels",mnist.test.labels)
    print_more("mnist.train.images",mnist.train.images)
    print_more("mnist.validation.images",mnist.validation.images)
    print_more("mnist.test.images",mnist.test.images)

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt