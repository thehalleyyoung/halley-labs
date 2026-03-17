//! Criterion benchmarks for the choreo-spatial crate.
//!
//! Tests R-tree insertion/query, GJK intersection, BVH construction,
//! and geometric transform operations at varying scales.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use choreo_spatial::{
    BVH, CollisionShape, GjkResult, RTree, RTreeConfig, RTreeEntry,
    gjk_intersection, gjk_distance,
    compose_transforms, invert_transform, transform_point, transform_aabb,
};
use choreo_types::geometry::{AABB, Point3, Sphere, Transform3D, Vector3, Quaternion};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_aabb(cx: f64, cy: f64, cz: f64, half: f64) -> AABB {
    AABB {
        min: Point3 { x: cx - half, y: cy - half, z: cz - half },
        max: Point3 { x: cx + half, y: cy + half, z: cz + half },
    }
}

fn make_sphere(cx: f64, cy: f64, cz: f64, r: f64) -> Sphere {
    Sphere {
        center: Point3 { x: cx, y: cy, z: cz },
        radius: r,
    }
}

fn make_identity_transform() -> Transform3D {
    Transform3D {
        translation: Vector3 { x: 0.0, y: 0.0, z: 0.0 },
        rotation: Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
        scale: Vector3 { x: 1.0, y: 1.0, z: 1.0 },
    }
}

fn make_transform(tx: f64, ty: f64, tz: f64) -> Transform3D {
    Transform3D {
        translation: Vector3 { x: tx, y: ty, z: tz },
        rotation: Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
        scale: Vector3 { x: 1.0, y: 1.0, z: 1.0 },
    }
}

// ---------------------------------------------------------------------------
// R-tree benchmarks
// ---------------------------------------------------------------------------

fn bench_rtree_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtree_insertion");
    for n in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut tree = RTree::new(RTreeConfig::default());
                for i in 0..n {
                    let x = (i as f64) * 0.1;
                    let entry = RTreeEntry::new(i as u64, make_aabb(x, 0.0, 0.0, 0.5));
                    tree.insert(entry);
                }
                black_box(&tree);
            });
        });
    }
    group.finish();
}

fn bench_rtree_range_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtree_range_query");
    for n in [100, 1_000, 10_000] {
        let mut tree = RTree::new(RTreeConfig::default());
        for i in 0..n {
            let x = (i as f64) * 0.1;
            tree.insert(RTreeEntry::new(i as u64, make_aabb(x, 0.0, 0.0, 0.5)));
        }
        let query_box = make_aabb(5.0, 0.0, 0.0, 2.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let results = tree.query(&query_box);
                black_box(results);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GJK benchmarks
// ---------------------------------------------------------------------------

fn bench_gjk_sphere_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("gjk_sphere_sphere");
    let a = CollisionShape::Sphere(make_sphere(0.0, 0.0, 0.0, 1.0));

    for dist in [0.5, 1.5, 3.0] {
        let b = CollisionShape::Sphere(make_sphere(dist, 0.0, 0.0, 1.0));
        group.bench_with_input(
            BenchmarkId::new("distance", format!("{:.1}", dist)),
            &b,
            |bench, shape_b| {
                bench.iter(|| black_box(gjk_intersection(&a, shape_b)));
            },
        );
    }
    group.finish();
}

fn bench_gjk_aabb_aabb(c: &mut Criterion) {
    let mut group = c.benchmark_group("gjk_aabb_aabb");
    let a = CollisionShape::Aabb(make_aabb(0.0, 0.0, 0.0, 1.0));

    for dist in [0.5, 1.5, 5.0] {
        let b = CollisionShape::Aabb(make_aabb(dist, 0.0, 0.0, 1.0));
        group.bench_with_input(
            BenchmarkId::new("distance", format!("{:.1}", dist)),
            &b,
            |bench, shape_b| {
                bench.iter(|| black_box(gjk_intersection(&a, shape_b)));
            },
        );
    }
    group.finish();
}

fn bench_gjk_distance(c: &mut Criterion) {
    let a = CollisionShape::Sphere(make_sphere(0.0, 0.0, 0.0, 1.0));
    let b = CollisionShape::Sphere(make_sphere(5.0, 0.0, 0.0, 1.0));
    c.bench_function("gjk_distance_separated", |bench| {
        bench.iter(|| black_box(gjk_distance(&a, &b)));
    });
}

// ---------------------------------------------------------------------------
// BVH benchmarks
// ---------------------------------------------------------------------------

fn bench_bvh_construction(c: &mut Criterion) {
    use choreo_spatial::bvh::BVHItem;

    let mut group = c.benchmark_group("bvh_construction");
    for n in [100, 1_000, 5_000] {
        let items: Vec<BVHItem<u64>> = (0..n)
            .map(|i| {
                let x = (i as f64) * 0.2;
                BVHItem::new(i as u64, make_aabb(x, 0.0, 0.0, 0.5))
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &items, |b, items| {
            b.iter(|| {
                let bvh = BVH::build(items.clone());
                black_box(bvh);
            });
        });
    }
    group.finish();
}

fn bench_bvh_query(c: &mut Criterion) {
    use choreo_spatial::bvh::BVHItem;

    let items: Vec<BVHItem<u64>> = (0..1_000)
        .map(|i| {
            let x = (i as f64) * 0.2;
            BVHItem::new(i as u64, make_aabb(x, 0.0, 0.0, 0.5))
        })
        .collect();
    let bvh = BVH::build(items);
    let query = make_aabb(10.0, 0.0, 0.0, 5.0);

    c.bench_function("bvh_query_1000", |b| {
        b.iter(|| {
            let hits = bvh.query(&query);
            black_box(hits);
        });
    });
}

// ---------------------------------------------------------------------------
// Transform benchmarks
// ---------------------------------------------------------------------------

fn bench_compose_transforms(c: &mut Criterion) {
    let a = make_transform(1.0, 2.0, 3.0);
    let b = make_transform(4.0, 5.0, 6.0);
    c.bench_function("compose_transforms", |bench| {
        bench.iter(|| black_box(compose_transforms(&a, &b)));
    });
}

fn bench_invert_transform(c: &mut Criterion) {
    let t = make_transform(1.0, 2.0, 3.0);
    c.bench_function("invert_transform", |bench| {
        bench.iter(|| black_box(invert_transform(&t)));
    });
}

fn bench_transform_point(c: &mut Criterion) {
    let t = make_transform(1.0, 2.0, 3.0);
    let p = Point3 { x: 7.0, y: 8.0, z: 9.0 };
    c.bench_function("transform_point", |bench| {
        bench.iter(|| black_box(transform_point(&t, &p)));
    });
}

fn bench_transform_aabb(c: &mut Criterion) {
    let t = make_transform(1.0, 2.0, 3.0);
    let aabb = make_aabb(0.0, 0.0, 0.0, 5.0);
    c.bench_function("transform_aabb", |bench| {
        bench.iter(|| black_box(transform_aabb(&t, &aabb)));
    });
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    spatial_benches,
    bench_rtree_insertion,
    bench_rtree_range_query,
    bench_gjk_sphere_sphere,
    bench_gjk_aabb_aabb,
    bench_gjk_distance,
    bench_bvh_construction,
    bench_bvh_query,
    bench_compose_transforms,
    bench_invert_transform,
    bench_transform_point,
    bench_transform_aabb,
);
criterion_main!(spatial_benches);
