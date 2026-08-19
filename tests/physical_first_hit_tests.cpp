#include <zombie/point_estimation/common.h>
#include <zombie/utils/fcpw_geometric_queries.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using zombie::FcpwNeumannBoundaryHandler;
using zombie::FcpwRobinBoundaryHandler;
using zombie::GeometricQueries;
using zombie::IntersectionPoint;
using zombie::Vector3;
using zombie::Vector3i;

struct QueryFixture {
    FcpwNeumannBoundaryHandler<3> handler;
    GeometricQueries<3> queries;
};

void buildFixture(QueryFixture& fixture,
                  const std::vector<Vector3>& positions,
                  const std::vector<Vector3i>& triangles,
                  float rayTMin)
{
    const auto keepAllSilhouettes = [](float, int) { return false; };
    fixture.handler.buildAccelerationStructure(
        positions, triangles, keepAllSilhouettes, true, false, false
    );
    const auto unitBranchWeight = [](float) { return 1.0f; };
    zombie::populateGeometricQueriesForNeumannBoundary<3>(
        fixture.handler, unitBranchWeight, fixture.queries, rayTMin
    );
}

bool targetPathPass()
{
    const Vector3 anchor(1000.0f, 0.0f, 0.0f);
    const Vector3 normal(1.0f, 0.0f, 0.0f);
    constexpr float inwardCosine = 4.0e-4f;
    const Vector3 direction(
        -inwardCosine,
        std::sqrt(1.0f - inwardCosine*inwardCosine),
        0.0f
    );
    constexpr float radius = 1.0f;
    constexpr float rayTMin = 1.0e-5f;
    constexpr float constructionHitT = 8.0e-1f;
    constexpr float anchorExtent = 2.0f;
    constexpr float curvedExtent = 3.0e-1f;

    std::vector<Vector3> positions{
        Vector3(1000.0f, -anchorExtent, -anchorExtent),
        Vector3(1000.0f,  anchorExtent, -anchorExtent),
        Vector3(1000.0f,  0.0f,         anchorExtent),
    };
    const Vector3 constructionHit = anchor + constructionHitT*direction;
    Vector3 tangentU(
        -constructionHit[1]/constructionHit[0], 1.0f, 0.0f
    );
    tangentU.normalize();
    const Vector3 tangentW(0.0f, 0.0f, 1.0f);
    positions.push_back(
        constructionHit + curvedExtent*tangentU + curvedExtent*tangentW
    );
    positions.push_back(
        constructionHit - curvedExtent*tangentU + curvedExtent*tangentW
    );
    positions.push_back(constructionHit - 2.0f*curvedExtent*tangentW);

    QueryFixture fixture;
    buildFixture(
        fixture, positions,
        {Vector3i(0, 1, 2), Vector3i(3, 4, 5)}, rayTMin
    );

    IntersectionPoint<3> intersection;
    const bool hit = fixture.queries.intersectReflectingBoundary(
        anchor, normal, direction, radius, true, intersection
    );
    if (!hit) return false;

    constexpr float expectedDistance = 0.83582442998886108f;
    constexpr float tolerance = 1.0e-6f;
    const Vector3 expectedPoint = anchor + intersection.dist*direction;
    const float pointDifference = (intersection.pt - expectedPoint).norm();
    const bool pass = intersection.dist > rayTMin
        && intersection.dist <= radius
        && std::fabs(intersection.dist - expectedDistance) <= tolerance
        && pointDifference <= tolerance;
    if (!pass) {
        std::cerr << "target distance=" << intersection.dist
                  << " point_difference=" << pointDifference << '\n';
    }
    return pass;
}

bool lowerBoundPass()
{
    constexpr float rayTMin = 1.0e-6f;
    constexpr float nearbyHitT = 2.0e-6f;
    constexpr float subThresholdHitT = 5.0e-7f;
    constexpr float radius = 1.0e-4f;
    constexpr float tolerance = 1.0e-9f;
    const Vector3 anchor(0.0f, 0.0f, 0.0f);
    const Vector3 normal(-1.0f, 0.0f, 0.0f);
    const Vector3 direction(1.0f, 0.0f, 0.0f);

    auto triangleAt = [](float x) {
        return std::vector<Vector3>{
            Vector3(x, -1.0f, -1.0f),
            Vector3(x,  1.0f, -1.0f),
            Vector3(x,  0.0f,  1.0f),
        };
    };

    std::vector<Vector3> nearbyPositions = triangleAt(0.0f);
    const std::vector<Vector3> nearbyTriangle = triangleAt(nearbyHitT);
    nearbyPositions.insert(
        nearbyPositions.end(), nearbyTriangle.begin(), nearbyTriangle.end()
    );
    QueryFixture nearbyFixture;
    buildFixture(
        nearbyFixture, nearbyPositions,
        {Vector3i(0, 1, 2), Vector3i(3, 4, 5)}, rayTMin
    );
    IntersectionPoint<3> nearbyIntersection;
    const bool nearbyHit = nearbyFixture.queries.intersectReflectingBoundary(
        anchor, normal, direction, radius, true, nearbyIntersection
    );

    std::vector<Vector3> rejectedPositions = triangleAt(0.0f);
    const std::vector<Vector3> rejectedTriangle = triangleAt(subThresholdHitT);
    rejectedPositions.insert(
        rejectedPositions.end(), rejectedTriangle.begin(), rejectedTriangle.end()
    );
    QueryFixture rejectedFixture;
    buildFixture(
        rejectedFixture, rejectedPositions,
        {Vector3i(0, 1, 2), Vector3i(3, 4, 5)}, rayTMin
    );
    IntersectionPoint<3> rejectedIntersection;
    const bool rejectedHit =
        rejectedFixture.queries.intersectReflectingBoundary(
            anchor, normal, direction, radius, true, rejectedIntersection
        );

    return nearbyHit
        && std::fabs(nearbyIntersection.dist - nearbyHitT) <= tolerance
        && !rejectedHit;
}

bool interiorControlPass()
{
    constexpr float rayTMin = 1.0e-6f;
    constexpr float hitT = 2.0e-5f;
    const std::vector<Vector3> positions{
        Vector3(hitT, -1.0f, -1.0f),
        Vector3(hitT,  1.0f, -1.0f),
        Vector3(hitT,  0.0f,  1.0f),
    };
    QueryFixture fixture;
    buildFixture(fixture, positions, {Vector3i(0, 1, 2)}, rayTMin);

    const Vector3 origin(0.0f, 0.0f, 0.0f);
    const Vector3 normal(-1.0f, 0.0f, 0.0f);
    const Vector3 direction(1.0f, 0.0f, 0.0f);
    IntersectionPoint<3> callbackIntersection;
    const bool callbackHit = fixture.queries.intersectReflectingBoundary(
        origin, normal, direction, 1.0e-4f, false, callbackIntersection
    );
    fcpw::Ray<3> directRay(origin, direction, 1.0e-4f);
    fcpw::Interaction<3> directInteraction;
    const bool directHit =
        fixture.handler.scene.getSceneData()->aggregate->intersect(
            directRay, directInteraction, false
        );

    return callbackHit == directHit
        && callbackHit
        && callbackIntersection.dist == directInteraction.d
        && (callbackIntersection.pt - directInteraction.p).norm() == 0.0f;
}

bool robinLowerBoundPass()
{
    constexpr float rayTMin = 1.0e-6f;
    constexpr float nearbyHitT = 2.0e-6f;
    const std::vector<Vector3> positions{
        Vector3(0.0f,       -1.0f, -1.0f),
        Vector3(0.0f,        1.0f, -1.0f),
        Vector3(0.0f,        0.0f,  1.0f),
        Vector3(nearbyHitT, -1.0f, -1.0f),
        Vector3(nearbyHitT,  1.0f, -1.0f),
        Vector3(nearbyHitT,  0.0f,  1.0f),
    };
    const std::vector<Vector3i> triangles{
        Vector3i(0, 1, 2), Vector3i(3, 4, 5)
    };
    const auto keepAllSilhouettes = [](float, int) { return false; };
    FcpwRobinBoundaryHandler<3> handler;
    handler.buildAccelerationStructure(
        positions, triangles, keepAllSilhouettes,
        {1.0f, 1.0f}, {1.0f, 1.0f}, false, false, false
    );
    GeometricQueries<3> queries;
    const auto unitBranchWeight = [](float) { return 1.0f; };
    zombie::populateGeometricQueriesForRobinBoundary<3>(
        handler, unitBranchWeight, queries, rayTMin
    );

    IntersectionPoint<3> intersection;
    const bool hit = queries.intersectReflectingBoundary(
        Vector3::Zero(), Vector3(-1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f), 1.0e-4f, true, intersection
    );
    return hit && std::fabs(intersection.dist - nearbyHitT) <= 1.0e-9f;
}

bool missEndpointPass()
{
    const Vector3 center(1000.0f, 0.0f, 0.0f);
    const Vector3 direction(-4.0e-4f, 0.99999988f, 0.0f);
    constexpr float radius = 1.0f;
    IntersectionPoint<3> intersection;
    zombie::setBallBoundaryIntersection<3>(
        center, direction, radius, intersection
    );

    return std::fabs((intersection.pt - center).norm() - radius) <= 1.0e-6f
        && intersection.dist == radius;
}

} // namespace

int main()
{
    const bool targetPass = targetPathPass();
    const bool lowerBound = lowerBoundPass();
    const bool interior = interiorControlPass();
    const bool robin = robinLowerBoundPass();
    const bool missEndpoint = missEndpointPass();
    const bool pass = targetPass && lowerBound && interior && robin && missEndpoint;

    std::cout << "{\"target_path\":" << (targetPass ? "true" : "false")
              << ",\"lower_bound\":" << (lowerBound ? "true" : "false")
              << ",\"interior\":" << (interior ? "true" : "false")
              << ",\"robin\":" << (robin ? "true" : "false")
              << ",\"miss_endpoint\":" << (missEndpoint ? "true" : "false")
              << ",\"result\":\"" << (pass ? "PASS" : "FAIL")
              << "\"}\n";
    return pass ? 0 : 1;
}
