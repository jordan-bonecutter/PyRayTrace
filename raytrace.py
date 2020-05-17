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
    # for every pixel, for bounce in bounces, ray cast
    for y in range(sheet.shape[0]):
      for x in range(sheet.shape[1]):
        samples = np.zeros((samples_per_pixel, 3))
        for sample in range(samples_per_pixel):
          # begin at the camera origin
          curr_point = camera.origin

          # each sample is moved around in the sub pixel for anti-aliasing
          # later i'll probably try to do stochastic sampling instead of
          # subpixel sampling
          curr_direction = camera.direction_vector(y+(sample/samples_per_pixel) - 0.5, x+(sample/samples_per_pixel) - 0.5, sheet.shape[0], sheet.shape[1])
          curr_weight = 0.5

          for bounce in range(bounce_limit):
            ray = Ray(curr_point, curr_direction)
            intersection = self.object_manager.closest_intersection(ray)

            # if there is no objects in the direction we're looking, we've hit nothing
            if not intersection:
              break

            # add color of reflected object
            samples[sample] += curr_weight*intersection.texture
            curr_weight *= 0.5

            # start next ray from intersection point
            curr_point = intersection.point

            # calculate bounce direction
            proj_d_n = np.dot(curr_direction, intersection.normal)*intersection.normal
            curr_direction = curr_direction - 2*proj_d_n

            # move away from the surface so that the next ray
            # doesn't hit it again
            curr_point += 0.00001*intersection.normal

        # set color
        sheet[y, x] = (1./3)*(samples[0] + samples[1] + samples[2])


class Ray:
  '''
  Data wrapper for 3d ray
  '''
  def __init__(self, origin, direction):
    self.origin = origin
    self.direction = direction


class Intersection:
  '''
  Holds information about bounce:
    point: bounce location
    normal: surface normal at bounce location
    texture: surface texture at bounce location
  '''
  def __init__(self, point, normal, texture):
    self.point = point
    self.normal = normal
    self.texture = texture


class DefaultObject:
  '''
  Object instance returned if no intersection is 
  detected for current ray
  '''
  def __init__(self):
    pass

  def get_color(self):
    return np.array([0, 0, 0])

  color = property(get_color)


class ObjectManager:
  '''
  Maintains all SceneObjects. Calculates closest intersection point.
  TODO: Add BVH functionality
  '''
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
  '''
  Camera object with origin and orientation.
  '''
  def __init__(self, origin, theta, phi, alpha, theta_range):
    # i have basic understanding for rotation with quaternions
    # (rotation plane is perpendicular to vector and rotates
    # according to right hand rule)
    self.origin = origin

    q1 = np.quaternion(math.cos(phi/2), 0, 0, math.sin(phi/2))
    q2 = np.quaternion(math.cos(-phi/2), 0, 0, math.sin(-phi/2))

    new_j = q1*np.quaternion(0, 0, 1, 0)*q2
    new_j.real = 0

    # I rotate along the new j so that there is no rotation about i.
    q3 = np.quaternion(math.cos(theta/2), 0, 0, 0) + math.sin(theta/2)*new_j
    q4 = np.quaternion(math.cos(theta/2), 0, 0, 0) - math.sin(theta/2)*new_j

    new_i = q3*(q1*np.quaternion(0, 1, 0, 0)*q2)*q4
    new_i.real = 0

    q5 = np.quaternion(math.cos(alpha/2), 0, 0, 0) + math.sin(alpha/2)*new_i
    q6 = np.quaternion(math.cos(alpha/2), 0, 0, 0) - math.sin(alpha/2)*new_i

    # the camera "looking" direction lies along i. The j direction
    # is right to the camera and the k direction is up.
    self.direction = (q5*(q3*(q1*np.quaternion(0, 1, 0, 0)*q2)*q4)*q6).imag
    self.right     = (q5*(q3*(q1*np.quaternion(0, 0, 1, 0)*q2)*q4)*q6).imag
    self.up        = (q5*(q3*(q1*np.quaternion(0, 0, 0, 1)*q2)*q4)*q6).imag

    self.theta_range = theta_range


  def set_orientation(self, theta, phi, alpha):
    q1 = np.quaternion(math.cos(phi/2), 0, 0, math.sin(phi/2))
    q2 = np.quaternion(math.cos(-phi/2), 0, 0, math.sin(-phi/2))

    new_j = q1*np.quaternion(0, 0, 1, 0)*q2
    new_j.real = 0

    q3 = np.quaternion(math.cos(theta/2), 0, 0, 0) + math.sin(theta/2)*new_j
    q4 = np.quaternion(math.cos(theta/2), 0, 0, 0) - math.sin(theta/2)*new_j

    new_i = q3*(q1*np.quaternion(0, 1, 0, 0)*q2)*q4
    new_i.real = 0

    q5 = np.quaternion(math.cos(alpha/2), 0, 0, 0) + math.sin(alpha/2)*new_i
    q6 = np.quaternion(math.cos(alpha/2), 0, 0, 0) - math.sin(alpha/2)*new_i

    # the camera "looking" direction lies along i. The j direction
    # is right to the camera and the k direction is up.
    self.direction = (q5*(q3*(q1*np.quaternion(0, 1, 0, 0)*q2)*q4)*q6).imag
    self.right     = (q5*(q3*(q1*np.quaternion(0, 0, 1, 0)*q2)*q4)*q6).imag
    self.up        = (q5*(q3*(q1*np.quaternion(0, 0, 0, 1)*q2)*q4)*q6).imag


  def move(self, origin):
    self.origin = origin


  def direction_vector(self, y, x, height, width):
    '''
    Get direction vector from camera to point on image plane
    '''
    l = height/(2*math.tan(self.theta_range/2)) 
    y = height/2 - y
    x = x - width/2
    direction = (l*self.direction) + (y*self.up) + (x*self.right)
    return direction/np.linalg.norm(direction)


class SceneObject:
  '''
  Base class for SceneObjects. Should not be instatiated
  '''
  def closest_intersection(self, ray):
    pass


class Plane(SceneObject):
  '''
  Plane scene object.
  '''
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
  '''
  Sphere SceneObject
  '''
  def __init__(self, origin, radius, color):
    self.origin = origin
    self.radius = radius
    self.color  = color

  def dist_to(self, point):
    return np.linalg.norm(self.origin - point) - self.radius

  # i got this method from wikipedia. it work p good
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
  
writer = cv.VideoWriter('rt_test.avi', cv.VideoWriter_fourcc(*'MJPG'), 24, (200, 200))

# hold scene objects in objects list
objects = []
objects.append(Sphere(np.array([0., 0., 70.]), 25, np.array([100, 0, 220])))
objects.append(Sphere(np.array([10., 10., 20.]), 12, np.array([220, 0, 100])))
objects.append(Plane(np.array([0., 0., 1.]), np.array([0, 0, -10.]), np.array([30, 30, 30])))

# create object manager from objects
om     = ObjectManager(objects)

# create camera
camera = Camera(np.array([50, 50, 50], dtype=np.float64), 0, 5*math.pi/4, 0, math.pi/1.5)
scene  = Scene(om, camera)
sheet  = np.zeros((200, 200, 3), dtype=np.uint8)
#scene.render(sheet)
#cv.imwrite('rttest.png', sheet)


for i in range(48):
  #s1_off = math.sin(math.tau*i/(24*1.5))
  #s2_off = math.sin(math.tau*i/24)
  s1_off = 0
  s2_off = 0
  objects[0].origin[0] += s1_off*5
  objects[1].origin[0] += s2_off*3

  scene.render(sheet)
  writer.write(sheet)
  sheet[:, :, :] = 0

  objects[0].origin[0] -= s1_off*10
  objects[1].origin[0] -= s2_off*5

  alpha = math.pi*i/48
  camera.set_orientation(0, 5*math.pi/4, alpha)
  camera.move(np.array([50-(0.9*i), 50-(0.9*i), 50]))


  print(i/(2*24))

