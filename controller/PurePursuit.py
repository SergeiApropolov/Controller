

# Borrowing some code from here: https://github.com/thomasfermi/Algorithms-for-Automated-Driving/blob/master/LICENSE

import numpy as np
import random

class PurePursuit:

    def __init__(self, global_path3d=None, lookahead=None, k=0.5):
        # global_path3d is a 4x4xN matrix, where each row is [x,y,z,1]
        self.global_path3d = global_path3d
        self.pose = None


        self.lookahead = lookahead
        self.k = k
        self.K_dd = 0.4
        self.wheel_base = 1.2
        self.min_lookahead = 1.5
        self.max_lookahead = 3.0
        if self.lookahead is not None:
            self.min_lookahead = self.lookahead
            self.max_lookahead = self.lookahead

    def set_global_path(self, global_path3d):

        self.global_path3d = global_path3d

    def update_pose(self, pose):

        # Pose is a 4x4 matrix
        self.pose = pose

        if self.global_path3d is None:
            return None

        # Transform global path into base frame based on pose inverse
        waypoints = np.matmul(np.linalg.inv(self.pose), self.global_path3d.T).T[:,:2]

        speed = 1.0 # m/s
        control_out = self.get_control(waypoints, speed)

        if control_out is None:
            return None
        
        (delta, alpha, track_point, segment_idx) = control_out

        # Transform the track point back into global frame
        track_point_global = np.matmul(self.pose, np.array([track_point[0], track_point[1], 0, 1]))[:3]

        return (delta, alpha, track_point_global, segment_idx)

        # TODO: output steering angle, steering angle, target point, target index

    
    def get_control(self, waypoints, speed):
        # transform x coordinates of waypoints such that coordinate origin is in rear wheel
        look_ahead_distance = np.clip(self.K_dd * speed, self.min_lookahead, self.max_lookahead)

        track_point, segment_idx = self.get_target_point(look_ahead_distance, waypoints)
        if track_point is None:
            return None

        alpha = np.arctan2(track_point[1], track_point[0])

        # Change the steer output with the lateral controller.
        delta = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)

        # undo transform to waypoints 
        return (delta, alpha, track_point, segment_idx)

    # From: https://github.com/thomasfermi/Algorithms-for-Automated-Driving/blob/master/code/solutions/control/get_target_point.py
    # Function from https://stackoverflow.com/a/59582674/2609987
    def circle_line_segment_intersection(self, circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
        """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
        :param circle_center: The (x, y) location of the circle center
        :param circle_radius: The radius of the circle
        :param pt1: The (x, y) location of the first point of the segment
        :param pt2: The (x, y) location of the second point of the segment
        :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
        :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
        :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.
        Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
        """

        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2)**.5
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

        if discriminant < 0:  # No intersection between circle and line
            return []
        else:  # There may be 0, 1, or 2 intersections with the segment
            intersections = [
                (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
                cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
                for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
            if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
                fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
                intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
            if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
                return [intersections[0]]
            else:
                return intersections

    def get_target_point(self, lookahead, polyline):
        """ Determines the target point for the pure pursuit controller
        
        Parameters
        ----------
        lookahead : float
            The target point is on a circle of radius `lookahead`
            The circle's center is (0,0)
        poyline: array_like, shape (M,2)
            A list of 2d points that defines a polyline.
        
        Returns:
        --------
        target_point: numpy array, shape (,2)
            Point with positive x-coordinate where the circle of radius `lookahead`
            and the polyline intersect. 
            Return None if there is no such point.  
            If there are multiple such points, return the one that the polyline
            visits first.
        """
        intersections = []
        segment_idx = -1
        for j in range(len(polyline)-1):
            pt1 = polyline[j]
            pt2 = polyline[j+1]
            intersections_in_segment = self.circle_line_segment_intersection((0,0), lookahead, pt1, pt2, full_line=False)
            intersections += intersections_in_segment
            if len(intersections_in_segment) > 0 and segment_idx == -1:
                segment_idx = j
        filtered = [p for p in intersections if p[0]>0]
        if len(filtered)==0:
            return None, None
        return filtered[0], segment_idx
