const skinVS = `#version 300 es
in vec4 a_POSITION;
in vec3 a_NORMAL;
in vec4 a_WEIGHTS_0;
in uvec4 a_JOINTS_0;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_world;
uniform sampler2D u_jointTexture;

out vec3 v_normal;

mat4 getBoneMatrix(uint jointNdx) {
  return mat4(
    texelFetch(u_jointTexture, ivec2(0, jointNdx), 0),
    texelFetch(u_jointTexture, ivec2(1, jointNdx), 0),
    texelFetch(u_jointTexture, ivec2(2, jointNdx), 0),
    texelFetch(u_jointTexture, ivec2(3, jointNdx), 0));
}

void main() {
  mat4 skinMatrix = getBoneMatrix(a_JOINTS_0[0]) * a_WEIGHTS_0[0] +
                    getBoneMatrix(a_JOINTS_0[1]) * a_WEIGHTS_0[1] +
                    getBoneMatrix(a_JOINTS_0[2]) * a_WEIGHTS_0[2] +
                    getBoneMatrix(a_JOINTS_0[3]) * a_WEIGHTS_0[3];
  mat4 world = u_world * skinMatrix;
  gl_Position = u_projection * u_view * world * a_POSITION;
  v_normal = mat3(world) * a_NORMAL;

  // for debugging .. see article
  //gl_Position = u_projection * u_view *  a_POSITION;
  //v_normal = a_NORMAL;
  //v_normal = a_WEIGHTS_0.xyz * 2. - 1.;
  //v_normal = vec3(a_JOINTS_0.xyz) / float(textureSize(u_jointTexture, 0).y - 1) * 2. - 1.;
}
`;
const fs = `#version 300 es
precision highp float;

in vec3 v_normal;

uniform vec4 u_diffuse;
uniform vec3 u_lightDirection;

out vec4 outColor;

void main () {
  vec3 normal = normalize(v_normal);
  float light = dot(u_lightDirection, normal) * .5 + .5;
  outColor = vec4(u_diffuse.rgb * light, u_diffuse.a);

  // for debugging .. see article
  //outColor = vec4(1, 0, 0, 1);
  //outColor = vec4(v_normal * .5 + .5, 1);
}
`;
const meshVS = `#version 300 es
in vec4 a_POSITION;
in vec3 a_NORMAL;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_world;

out vec3 v_normal;

void main() {
  gl_Position = u_projection * u_view * u_world * a_POSITION;
  v_normal = mat3(u_world) * a_NORMAL;
}
`;

function CreateShader(gl,vertCode,fragCode){

     function CreateShader(type,code){
        const shader = gl.createShader(type)
        gl.shaderSource(shader,code)
        gl.compileShader(shader)

        const message = gl.getShaderInfoLog(shader);
        if (message.length > 0) {
           /* message may be an error or a warning */
           throw message;
           }

        return shader
     }

        var vertShader = CreateShader(gl.VERTEX_SHADER,vertCode)
        var fragShader = CreateShader(gl.FRAGMENT_SHADER,fragCode);

        var program = gl.createProgram();
        gl.attachShader(program, vertShader);
        gl.attachShader(program, fragShader);
        gl.linkProgram(program);

   return {program}
   
}