#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// Compiled 4 ways by injecting exactly one MODE_* definition after #version.
// A distance is an exact 16-bit integer in an RGBA8 render target: R=high, G=low.
// 0 = unvisited; 1 = goal; 2,3,... = Manhattan steps + 1.
// 65535 is the largest representable distance; warn if it is reached.
uniform sampler2D uMaze;
uniform sampler2D uDistance;
uniform ivec2 uSize;
uniform ivec2 uGoal;
uniform float uTime;
out vec4 fragColor;

bool inMap(ivec2 p) {
    return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, uSize));
}

bool passable(ivec2 p) {
    return inMap(p) && texelFetch(uMaze, p, 0).r > 0.5;
}

float decodeDistance(vec4 packed) {
    return floor(packed.r * 255.0 + 0.5) * 256.0
         + floor(packed.g * 255.0 + 0.5);
}

float distanceAt(ivec2 p) {
    if (!inMap(p)) return 0.0;
    return decodeDistance(texelFetch(uDistance, p, 0));
}

vec4 encodeDistance(float d) {
    float hi = floor(d / 256.0);
    float lo = d - hi * 256.0;
    return vec4(hi / 255.0, lo / 255.0, 0.0, 1.0);
}

float minNeighbor(ivec2 p) {
    float best = 65536.0;
    float d;
    d = distanceAt(p + ivec2( 1, 0)); if (d > 0.0) best = min(best, d);
    d = distanceAt(p + ivec2(-1, 0)); if (d > 0.0) best = min(best, d);
    d = distanceAt(p + ivec2( 0, 1)); if (d > 0.0) best = min(best, d);
    d = distanceAt(p + ivec2( 0,-1)); if (d > 0.0) best = min(best, d);
    return best;
}

#ifdef MODE_WAVE
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (!passable(p)) {
        fragColor = vec4(0.0);
        return;
    }
    float oldValue = distanceAt(p);
    if (oldValue > 0.0) {
        fragColor = encodeDistance(oldValue);
    } else if (all(equal(p, uGoal))) {
        fragColor = encodeDistance(1.0);
    } else {
        float d = minNeighbor(p);
        fragColor = d < 65535.0 ? encodeDistance(d + 1.0) : vec4(0.0);
    }
}
#endif

#ifdef MODE_FRONTIER
// Run under gl.ANY_SAMPLES_PASSED. No texture data is read back.
// Discard everything unless some unreached passable pixel can be visited next.
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (!passable(p) || distanceAt(p) > 0.0) discard;
    if (all(equal(p, uGoal)) || minNeighbor(p) < 65535.0) {
        fragColor = vec4(1.0);
    } else {
        discard;
    }
}
#endif

#ifdef MODE_AGENTS
uniform sampler2D uAgents;
uniform ivec2 uAgentTexSize;

ivec2 getPos(vec4 data) {
    int x = int(floor(data.r * 255.0 + 0.5)) * 256
          + int(floor(data.g * 255.0 + 0.5));
    int y = int(floor(data.b * 255.0 + 0.5)) * 256
          + int(floor(data.a * 255.0 + 0.5));
    return ivec2(x,y);
}

vec4 packPos(ivec2 pos) {
    return vec4(float(pos.x / 256), float(pos.x % 256),
                float(pos.y / 256), float(pos.y % 256)) / 255.0;
}

void considerMove(inout ivec2 bestPos, inout float bestValue, ivec2 candidate) {
    float d = distanceAt(candidate);
    if (d > 0.0 && d < bestValue) {
        bestValue = d;
        bestPos = candidate;
    }
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 agent = getPos(texelFetch(uAgents, p, 0));
    float here = distanceAt(agent);
    // Leave agents in place while the field is still unreached.
    if (here <= 1.0) {
        fragColor = packPos(agent);
        return;
    }
    ivec2 bestPos = agent;
    float bestValue = here;
    considerMove(bestPos, bestValue, agent + ivec2(0,-1));
    considerMove(bestPos, bestValue, agent + ivec2(0, 1));
    considerMove(bestPos, bestValue, agent + ivec2(-1,0));
    considerMove(bestPos, bestValue, agent + ivec2( 1,0));
    fragColor = packPos(bestPos);
}
#endif

#ifdef MODE_DISPLAY
uniform vec2 uViewport;
void main() {
    // gl_FragCoord has bottom-left origin; our maze coordinates have top-left.
    ivec2 screen = ivec2(gl_FragCoord.xy);
    ivec2 p = ivec2(floor(gl_FragCoord.xy / uViewport * vec2(uSize)));
    p = clamp(p, ivec2(0), uSize - ivec2(1));
    p.y = uSize.y - 1 - p.y;
    if (!passable(p)) {
        fragColor = vec4(0.015, 0.018, 0.030, 1.0);
        return;
    }
    float distanceValue = distanceAt(p);
    if (distanceValue < 1.0) {
        fragColor = vec4(0.105, 0.12, 0.17, 1.0);
    } else {
        float pulse = 0.5 + 0.5 * sin(distanceValue * 0.045 - uTime * 0.002);
        vec3 cyan = vec3(0.03, 0.85, 0.95);
        vec3 violet = vec3(0.64, 0.14, 0.95);
        fragColor = vec4(mix(cyan, violet, pulse) * 0.85, 1.0);
    }
}
#endif
