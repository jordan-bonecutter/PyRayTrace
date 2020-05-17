#
#
#
#

import numpy as np
import quaternion
import math
import cv2 as cv
import random

class Scene:
  def __init__(self, object_manager, camera=None):
    self.object_manager = object_manager 
    if camera is None:
      camera = Camera()
    self.camera = camera


  def render(self, sheet, samples_per_pixel=3, bounce_limit=3):
    '''
    Use ray casting to render out scene objects
    '''
    # for every pixel, ray march until bounce
    for y in range(sheet.shape[0]):
      for x in range(sheet.shape[1]):
        samples = np.zeros((samples_per_pixel, 3))
        for sample in range(samples_per_pixel):
          curr_point = camera.origin
          curr_direction = camera.direction_vector(y+(sample/samples_per_pixel) - 0.5, x+(sample/samples_per_pixel) - 0.5, sheet.shape[0], sheet.shape[1])
          curr_weight = 0.5

          for bounce in range(bounce_limit):
            ray = Ray(curr_point, curr_direction)
            intersection = self.object_manager.closest_intersection(ray)

            if not intersection:
              break

            samples[sample] += curr_weight*intersection.texture
            curr_weight *= 0.5
            curr_point = intersection.point
            proj_d_n = np.dot(curr_direction, intersection.normal)*intersection.normal
            curr_direction = curr_direction - 2*proj_d_n
            curr_point += 0.001*curr_direction

        sheet[y, x] = (1./3)*(samples[0] + samples[1] + samples[2])

class Ray:
  def __init__(self, origin, direction):
    self.origin = origin
    self.direction = direction


class Intersection:
  def __init__(self, point, normal, texture):
    self.point = point
    self.normal = normal
    self.texture = texture


class DefaultObject:
  def __init__(self):
    pass

  def get_color(self):
    return np.array([0, 0, 0])

  color = property(get_color)


class ObjectManager:
  def __init__(self, scene_objects):
    self.scene_objects = scene_objects

  def min_dist(self, point):
    ret = 1e300
    for scene_object in self.scene_objects:
      ret = min(ret, scene_object.dist_to(point))

    return ret

  def closest_intersection(self, ray):
    mindist = 1e300
    minintersect = None
    for scene_object in self.scene_objects:
      didintersect, intersection = scene_object.closest_intersection(ray)
      if didintersect:
        dist = np.linalg.norm(intersection.point - ray.origin)
        if dist < mindist:
          mindist = dist
          minintersect = intersection

    if minintersect is None:
      return None

    else:
      return minintersect


class Camera:
  def __init__(self, origin, theta, phi, theta_range):
    self.origin = origin
    q1 = np.quaternion(math.cos(phi/2), 0, 0, math.sin(phi/2))
    q2 = np.quaternion(math.cos(-phi/2), 0, 0, math.sin(-phi/2))
    #q3 = np.quaternion(math.cos(theta/2), 0,  math.sin(theta/2), 0)
    #q4 = np.quaternion(math.cos(-theta/2), 0, math.sin(-theta/2), 0)
    new_j = q1*np.quaternion(0, 0, 1, 0)*q2
    new_j.real = 0
    #q3 = np.quaternion(math.cos(theta/2), 0, 0, 0) + math.sin(theta/2)*new_j
    #q4 = np.quaternion(math.cos(theta/2), 0, 0, 0) - math.sin(theta/2)*new_j
    q3 = np.quaternion(1, 0, 0, 0)
    q4 = np.quaternion(1, 0, 0, 0)
    self.direction = (q3*(q1*np.quaternion(0, 1, 0, 0)*q2)*q4).imag
    self.right     = (q3*(q1*np.quaternion(0, 0, 1, 0)*q2)*q4).imag
    self.up        = (q3*(q1*np.quaternion(0, 0, 0, 1)*q2)*q4).imag

    self.theta_range = theta_range

  def direction_vector(self, y, x, height, width):
    l = height/(2*math.tan(self.theta_range/2)) 
    y = height/2 - y
    x = x - width/2
    direction = (l*self.direction) + (y*self.up) + (x*self.right)
    return direction/np.linalg.norm(direction)


  def move_forward(self, dist):
    self.origin += self.direction*dist


class SceneObject:
  def closest_intersection(self, ray):
    pass


class Plane(SceneObject):
  def __init__(self, normal, point, color):
    self.normal = normal
    self.point  = point
    self.color  = color

  def closest_intersection(self, ray):
    d = np.dot(self.point - ray.origin, self.normal)/np.dot(ray.direction, self.normal)
    if d < 0:
      return False, None
    else:
      return True, Intersection(ray.origin + d*ray.direction, self.normal, self.color)


class Sphere(SceneObject):
  def __init__(self, origin, radius, color):
    self.origin = origin
    self.radius = radius
    self.color  = color

  def dist_to(self, point):
    return np.linalg.norm(self.origin - point) - self.radius

  def closest_intersection(self, ray):
    determinant = np.dot(ray.direction, ray.origin - self.origin)**2 - \
                  np.linalg.norm(ray.origin - self.origin)**2 + self.radius**2
    if determinant < 0:
      return False, None

    else:
      determinant = math.sqrt(determinant)
      pref = -np.dot(ray.direction, ray.origin - self.origin)
      d1 = pref + determinant
      d2 = pref - determinant
      if d1 > 0:
        if d2 > 0:
          if d1 < d2:
            point = ray.origin + d1*ray.direction
            normal = point - self.origin
            normal /= np.linalg.norm(normal)
            return True, Intersection(point, normal, self.color)
          else:
            point = ray.origin + d2*ray.direction
            normal = point - self.origin
            normal /= np.linalg.norm(normal)
            return True, Intersection(point, normal, self.color)
        else:
          point = ray.origin + d1*ray.direction
          normal = point - self.origin
          normal /= np.linalg.norm(normal)
          return True, Intersection(point, normal, self.color)
      elif d2 > 0:
        point = ray.origin + d2*ray.direction
        normal = point - self.origin
        normal /= np.linalg.norm(normal)
        return True, Intersection(point, normal, self.color)
      else:
        return False, None
  
#writer = cv.VideoWriter('rt_test.mkv', cv.VideoWriter_fourcc(*'x264'), 24, (400, 400))

objects = []
objects.append(Sphere(np.array([0., 0., 70.]), 25, np.array([100, 0, 220])))
objects.append(Sphere(np.array([10., 10., 20.]), 12, np.array([220, 0, 100])))
objects.append(Plane(np.array([0., 0., 1.]), np.array([0, 0, -10.]), np.array([30, 30, 30])))
om     = ObjectManager(objects)
camera = Camera(np.array([50, 50, 50], dtype=np.float64), -math.pi/4, 5*math.pi/4, math.pi/1.5)
scene  = Scene(om, camera)
sheet  = np.zeros((200, 200, 3), dtype=np.uint8)
scene.render(sheet)
cv.imwrite('rttest.png', sheet)

'''
for frame in range(2):
  s1_off = math.sin(math.tau*frame/(24*1.5))
  s2_off = math.sin(math.tau*frame/24)
  objects[0].origin[0] += s1_off*10
  objects[1].origin[0] += s2_off*5

  scene.render(sheet)
  writer.write(sheet)
  sheet[:, :, :] = 0

  objects[0].origin[0] -= s1_off*10
  objects[1].origin[0] -= s2_off*5
  print(frame/(3*24))
'''

