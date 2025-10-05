#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// Determines if a point is inside a polygon (works for non-convex polygons).
///
/// This function uses the ray-casting algorithm (even-odd rule).
/// It casts a ray from the point in a horizontal direction and counts how many times it intersects with the polygon edges.
/// If the number of intersections is odd, the point is inside; if even, it is outside.
///
/// # Arguments
/// * `point` - The point to check.
/// * `polygon` - An array of vertices of the polygon.
///
/// # Returns
/// * Returns `true` if the point is inside the polygon, otherwise `false`.
pub fn is_inside_polygon(point: Point, polygon: &[Point]) -> bool {
    let num_vertices = polygon.len();
    if num_vertices < 3 {
        return false;
    }

    let mut is_inside = false;
    let mut j = num_vertices - 1;
    for i in 0..num_vertices {
        let pi = polygon[i];
        let pj = polygon[j];

        // Check if the point is between the y-coordinates of the vertices and to the left of the edge.
        if (pi.y > point.y) != (pj.y > point.y) {
            let x_intersection = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x;
            if point.x < x_intersection {
                is_inside = !is_inside;
            }
        }
        j = i;
    }

    is_inside
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_inside_polygon() {
        let square = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 0.0, y: 2.0 },
        ];

        // Point inside
        assert!(is_inside_polygon(Point { x: 1.0, y: 1.0 }, &square));

        // Point on edge (behavior can be ambiguous with ray-casting, so not tested)
        // assert!(is_inside_polygon(Point { x: 1.0, y: 0.0 }, &square));
        // assert!(is_inside_polygon(Point { x: 2.0, y: 1.0 }, &square));

        // Point on vertex (behavior can be ambiguous with ray-casting)
        // assert!(is_inside_polygon(Point { x: 0.0, y: 0.0 }, &square));

        // Point outside
        assert!(!is_inside_polygon(Point { x: -1.0, y: 1.0 }, &square));
        assert!(!is_inside_polygon(Point { x: 1.0, y: 3.0 }, &square));
        assert!(!is_inside_polygon(Point { x: 3.0, y: 1.0 }, &square));

        // Test with a triangle
        let triangle = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 3.0, y: 1.0 },
            Point { x: 1.0, y: 3.0 },
        ];
        assert!(is_inside_polygon(Point { x: 1.5, y: 1.5 }, &triangle));
        assert!(!is_inside_polygon(Point { x: 0.5, y: 2.5 }, &triangle));

        // Test with a non-convex polygon
        let u_shape = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 5.0, y: 0.0 },
            Point { x: 5.0, y: 5.0 },
            Point { x: 3.0, y: 5.0 },
            Point { x: 3.0, y: 2.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 2.0, y: 5.0 },
            Point { x: 0.0, y: 5.0 },
        ];

        // Inside the "U"
        assert!(is_inside_polygon(Point { x: 1.0, y: 1.0 }, &u_shape));
        assert!(is_inside_polygon(Point { x: 4.0, y: 4.0 }, &u_shape));

        // In the "hollow" part of the "U"
        assert!(!is_inside_polygon(Point { x: 2.5, y: 3.0 }, &u_shape));

        // Outside
        assert!(!is_inside_polygon(Point { x: -1.0, y: 1.0 }, &u_shape));
        assert!(!is_inside_polygon(Point { x: 6.0, y: 3.0 }, &u_shape));
    }
}