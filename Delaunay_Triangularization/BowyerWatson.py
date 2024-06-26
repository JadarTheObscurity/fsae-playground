"""
BowyerWatson 
function BowyerWatson (pointList)
    // pointList is a set of coordinates defining the points to be triangulated
    triangulation := empty triangle mesh data structure
    add super-triangle to triangulation // must be large enough to completely contain all the points in pointList
    for each point in pointList do // add all the points one at a time to the triangulation
        badTriangles := empty set
        for each triangle in triangulation do // first find all the triangles that are no longer valid due to the insertion
            if point is inside circumcircle of triangle
                add triangle to badTriangles
        polygon := empty set
        for each triangle in badTriangles do // find the boundary of the polygonal hole
            for each edge in triangle do
                if edge is not shared by any other triangles in badTriangles
                    add edge to polygon
        for each triangle in badTriangles do // remove them from the data structure
            remove triangle from triangulation
        for each edge in polygon do // re-triangulate the polygonal hole
            newTri := form a triangle from edge to point
            add newTri to triangulation
    for each triangle in triangulation // done inserting points, now clean up
        if triangle contains a vertex from original super-triangle
            remove triangle from triangulation
    return triangulation
"""
import numpy as np

def BowyerWatson(points):
    triangulation = []
    super_triangle = Triangle((0, 100), (-85, -50), (85, -50))
    triangulation.append(super_triangle)
    for p in points:
        bad_triangle = []
        for t in triangulation:
            if t.in_circumcircle(p):
                bad_triangle.append(t)
        polygon = []
        for t in bad_triangle:
            for edge in t.edges:
                if not any((edge == e and t != tri for tri in bad_triangle for e in tri.edges)):
                    polygon.append(edge)
            triangulation.remove(t)
        for edge in polygon:
            triangulation.append(Triangle(p, edge.p1, edge.p2))
    remove_triangles = []
    for t in triangulation:
        if any(np.all(np.equal(p1, p2)) for p1 in t.points for p2 in super_triangle.points):
            remove_triangles.append(t)
    for t in remove_triangles:
        triangulation.remove(t)
    return triangulation

def filtered_BowyerWatson(points):
    triangulation = BowyerWatson(points)
    filtered_triangulation = []
    for t in triangulation:
        # find the smallest angle in the triangle
        angles = []
        for i in range(3):
            first_vector = t.points[(i+1)%3] - t.points[i]
            second_vector = t.points[(i+1)%3] - t.points[(i+2)%3]
            cos_theta = np.dot(first_vector, second_vector) / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
            theta = np.arccos(cos_theta)
            angles.append(theta)
        min_angle = max(angles)
        if min_angle < 120.0 * np.pi / 180:
            # print(min_angle, np.rad2deg(min_angle))
            filtered_triangulation.append(t)
    return filtered_triangulation



class Triangle:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.points = [p1, p2, p3]
        self.edges = [ 
            Edge(p1, p2),
            Edge(p2, p3),
            Edge(p1, p3)
        ]
        self.center, self.r = self.get_circumcircle()
    
    def in_circumcircle(self, p):
        d = np.hypot(p[0]-self.center[0], p[1]-self.center[1])
        return d < self.r
        
    def get_circumcircle(self):
        """
        points: [(x1, y1), (x2, y2), (x3, y3)]
        returns: [(xc, yc), r]
        """
        x1, y1 = self.p1 
        x2, y2 = self.p2 
        x3, y3 = self.p3 

        if abs(x2-x1) < 1e-5:
            m1 = (y2 - y1) / 1e-5
        else:
            m1 = (y2 - y1) / (x2-x1)

        if abs(x3-x2) < 1e-5:
            m2 = (y3 - y2) / 1e-5
        else:
            m2 = (y3 - y2) / (x3-x2)

        xa = (x1+x2) / 2
        ya = (y1+y2) / 2
        xb = (x2+x3) / 2
        yb = (y2+y3) / 2

        if m1 == m2:
            return None, None

        xc = 1 / (m2-m1) * (m2 * xa - m1 * xb + m1 * m2 * (ya-yb))
        if abs(m1) > abs(m2):
            yc = -1 / m1 * (xc - xa) + ya
        else:
            yc = -1 / m2 * (xc - xb) + yb
        r = np.hypot(xc-x1, yc-y1)
        return (xc, yc), r



class Edge:
    def __init__(self, p1, p2):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
    
    def __eq__(self, e2):
        return (np.all(np.equal(self.p1, e2.p1)) and np.all(np.equal(self.p2, e2.p2))) or (np.all(np.equal(self.p1, e2.p1)) and np.all(np.equal(self.p2, e2.p2)))




if __name__ == '__main__':
    points = np.array([[1, 1], [-1, 1], [1, 2], [-1, 2], [1.1, 3], [-0.9, 2.9], [1.7, 4.7], [-0.5, 4.3]])
    triangulation = BowyerWatson(points)
    edges = []
    for tri in triangulation:
        for e in tri.edges:
            if e not in edges:
                edges.append(e)
    print(len(edges))
    mid_points = np.array([(e.p1 + e.p2) / 2 for e in edges])

    import matplotlib.pyplot as plt
    for t in triangulation:
        plt.plot([t.p1[0], t.p2[0], t.p3[0], t.p1[0]], [t.p1[1], t.p2[1], t.p3[1], t.p1[1]], 'b-')
    plt.plot([p[0] for p in points], [p[1] for p in points], 'ro')
    plt.scatter(mid_points[:,0], mid_points[:, 1])
    plt.show()