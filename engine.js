var {mat4,vec3} = glMatrix

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
 // var name = 'default'

  //if(SHADERCACHE[name]){
  //   return SHADERCACHE[name]
  //}

   //var vertCode = await get('/shaders/'+name+'.vert')
    //    var fragCode = await get('/shaders/'+name+'.frag')

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

  //SHADERCACHE[name] = {program}

   return {program}
   
}

async function main() {
  // Get A WebGL context
  /** @type {HTMLCanvasElement} */
  const canvas = document.querySelector("#canvas");
  const gl = canvas.getContext("webgl2");
  if (!gl) {
    return;
  }

  // Specify the locations of the attributes so they'll
  // match across programs
  const programOptions = {
    attribLocations: {
      a_POSITION: 0,
      a_NORMAL: 1,
      a_WEIGHTS_0: 2,
      a_JOINTS_0: 3,
    },
  };

  // compiles and links the shaders, looks up attribute and uniform locations
  //const skinProgramInfo = twgl.createProgramInfo(gl, [skinVS, fs], programOptions);
  const skinProgramInfo = CreateShader(gl, skinVS, fs);
  //const meshProgramInfo = twgl.createProgramInfo(gl, [meshVS, fs], programOptions);
  const meshProgramInfo = CreateShader(gl, meshVS, fs);

  class Skin {
    constructor(joints, inverseBindMatrixData) {
      this.joints = joints;
      this.inverseBindMatrices = [];
      this.jointMatrices = [];
      // allocate enough space for one matrix per joint
      this.jointData = new Float32Array(joints.length * 16);
      // create views for each joint and inverseBindMatrix
      for (let i = 0; i < joints.length; ++i) {
        this.inverseBindMatrices.push(new Float32Array(
            inverseBindMatrixData.buffer,
            inverseBindMatrixData.byteOffset + Float32Array.BYTES_PER_ELEMENT * 16 * i,
            16));
        this.jointMatrices.push(new Float32Array(
            this.jointData.buffer,
            Float32Array.BYTES_PER_ELEMENT * 16 * i,
            16));
      }
      // create a texture to hold the joint matrices
      this.jointTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.jointTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    update(node) {
      //const globalWorldInverse = m4.inverse(node.worldMatrix);
      let globalWorldInverse = mat4.create()
      mat4.invert(globalWorldInverse, node.worldMatrix);
      // go through each joint and get its current worldMatrix
      // apply the inverse bind matrices and store the
      // entire result in the texture
      for (let j = 0; j < this.joints.length; ++j) {
        const joint = this.joints[j];
        const dst = this.jointMatrices[j];
        //m4.multiply(globalWorldInverse, joint.worldMatrix, dst);
        mat4.multiply(dst, joint.worldMatrix, globalWorldInverse)
        //m4.multiply(dst, this.inverseBindMatrices[j], dst);
        mat4.multiply(dst, dst, this.inverseBindMatrices[j])
      }
      gl.bindTexture(gl.TEXTURE_2D, this.jointTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, this.joints.length, 0,
                    gl.RGBA, gl.FLOAT, this.jointData);
    }
  }

  class TRS {
    constructor(position = [0, 0, 0], rotation = [0, 0, 0, 1], scale = [1, 1, 1]) {
      this.position = position;
      this.rotation = rotation;
      this.scale = scale;
    }
    getMatrix(dst) {
      dst = dst || new Float32Array(16);
      //m4.compose(this.position, this.rotation, this.scale, dst);
      mat4.fromRotationTranslationScale(dst, this.rotation, this.position, this.scale)
      return dst;
    }
  }

  class Node {
    constructor(source, name) {
      this.name = name;
      this.source = source;
      this.parent = null;
      this.children = [];
      this.localMatrix = mat4.create()
      mat4.identity(this.localMatrix);
      this.worldMatrix = mat4.create()
      mat4.identity(this.worldMatrix);
      this.drawables = [];
    }
    setParent(parent) {
      if (this.parent) {
        this.parent._removeChild(this);
        this.parent = null;
      }
      if (parent) {
        parent._addChild(this);
        this.parent = parent;
      }
    }
    updateWorldMatrix(parentWorldMatrix) {
      const source = this.source;
      if (source) {
        source.getMatrix(this.localMatrix);
      }

      if (parentWorldMatrix) {
        // a matrix was passed in so do the math
        ///m4.multiply(parentWorldMatrix, this.localMatrix, this.worldMatrix);
        mat4.multiply(this.worldMatrix, parentWorldMatrix, this.localMatrix);
      } else {
        // no matrix was passed in so just copy local to world
        //m4.copy(this.localMatrix, this.worldMatrix);
        mat4.copy(this.worldMatrix, this.localMatrix)
      }

      // now process all the children
      const worldMatrix = this.worldMatrix;
      for (const child of this.children) {
        child.updateWorldMatrix(worldMatrix);
      }
    }
    traverse(fn) {
      fn(this);
      for (const child of this.children) {
        child.traverse(fn);
      }
    }
    _addChild(child) {
      this.children.push(child);
    }
    _removeChild(child) {
      const ndx = this.children.indexOf(child);
      this.children.splice(ndx, 1);
    }
  }

  class SkinRenderer {
    constructor(mesh, skin) {
      this.mesh = mesh;
      this.skin = skin;
    }
    render(node, projection, view, sharedUniforms) {
      const {skin, mesh} = this;
      skin.update(node);
      gl.useProgram(skinProgramInfo.program);
      for (const primitive of mesh.primitives) {
        gl.bindVertexArray(primitive.vao);
        var uni = uniformsSetter(gl,skinProgramInfo)
        uni.set({
          u_projection: projection,
          u_view: view,
          u_world: node.worldMatrix,
          u_jointTexture: skin.jointTexture,
          u_numJoints: skin.joints.length,
        })
        uni.set(primitive.material.uniforms)
        uni.set(sharedUniforms)
        /*twgl.setUniforms(skinProgramInfo, {
          u_projection: projection,
          u_view: view,
          u_world: node.worldMatrix,
          u_jointTexture: skin.jointTexture,
          u_numJoints: skin.joints.length,
        }, primitive.material.uniforms, sharedUniforms);*/
        //twgl.drawBufferInfo(gl, primitive.bufferInfo);
        //gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, primitive.bufferInfo.indices);
        //gl.bindVertexArray(primitive.vao);
            gl.drawElements(gl.TRIANGLES, primitive.bufferInfo.numElements, gl.UNSIGNED_SHORT, 0);
      }
    }
  }

  class MeshRednerer {
    constructor(mesh) {
      this.mesh = mesh;
    }
    render(node, projection, view, sharedUniforms) {
      const {mesh} = this;
      gl.useProgram(meshProgramInfo.program);
      for (const primitive of mesh.primitives) {
        gl.bindVertexArray(primitive.vao);
        var uni = uniformsSetter(gl,skinProgramInfo)
        uni.set({
          u_projection: projection,
          u_view: view,
          u_world: node.worldMatrix,
        })
        uni.set(primitive.material.uniforms)
        uni.set(sharedUniforms)
        /*twgl.setUniforms(meshProgramInfo, {
          u_projection: projection,
          u_view: view,
          u_world: node.worldMatrix,
        }, primitive.material.uniforms, sharedUniforms);
        twgl.drawBufferInfo(gl, primitive.bufferInfo);*/
        gl.drawElements(gl.TRIANGLES, primitive.bufferInfo.numElements, gl.UNSIGNED_SHORT, 0);
      }
    }
  }

  function throwNoKey(key) {
    throw new Error(`no key: ${key}`);
  }
  function NumSize(val){
    return {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16
    }[val]
}

  const accessorTypeToNumComponentsMap = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16,
  };

  function accessorTypeToNumComponents(type) {
    return accessorTypeToNumComponentsMap[type] || throwNoKey(type);
  }
  function glTypedArray(val){
    return {
        '5120': Int8Array,
        '5121': Uint8Array,
        '5122': Int16Array,
        '5123': Uint16Array,
        '5124': Int32Array,
        '5125': Uint32Array,
        '5126': Float32Array
    }[val]
}
  const glTypeToTypedArrayMap = {
    '5120': Int8Array,    // gl.BYTE
    '5121': Uint8Array,   // gl.UNSIGNED_BYTE
    '5122': Int16Array,   // gl.SHORT
    '5123': Uint16Array,  // gl.UNSIGNED_SHORT
    '5124': Int32Array,   // gl.INT
    '5125': Uint32Array,  // gl.UNSIGNED_INT
    '5126': Float32Array, // gl.FLOAT
  };

  // Given a GL type return the TypedArray needed
  function glTypeToTypedArray(type) {
    return glTypeToTypedArrayMap[type] || throwNoKey(type);
  }

  // given an accessor index return both the accessor and
  // a TypedArray for the correct portion of the buffer
  function getAccessorTypedArrayAndStride(gl, gltf, accessorIndex) {
    const accessor = gltf.accessors[accessorIndex];
    const bufferView = gltf.bufferViews[accessor.bufferView];
    const TypedArray = glTypeToTypedArray(accessor.componentType);
    const buffer = gltf.buffers[bufferView.buffer];
    return {
      accessor,
      array: new TypedArray(
          buffer,
          bufferView.byteOffset + (accessor.byteOffset || 0),
          accessor.count * accessorTypeToNumComponents(accessor.type)),
      stride: bufferView.byteStride || 0,
    };
  }

  // Given an accessor index return a WebGLBuffer and a stride
  function getAccessorAndWebGLBuffer(gl, gltf, accessorIndex) {
    const accessor = gltf.accessors[accessorIndex];
    const bufferView = gltf.bufferViews[accessor.bufferView];
    if (!bufferView.webglBuffer) {
      const buffer = gl.createBuffer();
      const target = bufferView.target || gl.ARRAY_BUFFER;
      const arrayBuffer = gltf.buffers[bufferView.buffer];
      const data = new Uint8Array(arrayBuffer, bufferView.byteOffset, bufferView.byteLength);
      gl.bindBuffer(target, buffer);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      bufferView.webglBuffer = buffer;
    }
    return {
      accessor,
      buffer: bufferView.webglBuffer,
      stride: bufferView.stride || 0,
    };
  }


  function createVAOFromBufferInfo(gl, skinProgramInfo, primitivebufferInfo){

    console.log(skinProgramInfo)
    console.log(primitivebufferInfo)

    var vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    gl.useProgram(skinProgramInfo.program)

    for(let key of Object.keys(primitivebufferInfo.attribs)){
      gl.bindBuffer(gl.ARRAY_BUFFER, primitivebufferInfo.attribs[key].buffer);
      const attribute = gl.getAttribLocation(skinProgramInfo.program, key)
      /*if(attribute>-1){
        console.log(key)
         gl.vertexAttribPointer(attribute, primitivebufferInfo.attribs[key].numComponents, 
          gl.FLOAT, false,0,0)
         gl.enableVertexAttribArray(attribute)
      }*/
      if(attribute>-1){
        console.log(key)
        if(key=='a_JOINTS_0'){
          gl.vertexAttribIPointer(attribute, primitivebufferInfo.attribs[key].numComponents, 
          gl.UNSIGNED_SHORT, false,0,0)
        }else{
          gl.vertexAttribPointer(attribute, primitivebufferInfo.attribs[key].numComponents, 
            gl.FLOAT, false,0,0)
        }
        gl.enableVertexAttribArray(attribute)
      }
    }

    if(primitivebufferInfo.indices){
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, primitivebufferInfo.indices);
    }

    gl.bindVertexArray(null);

    return vao

  }

  function uniformsSetter(gl,shader){
       
    let sampler=0
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D,null)
  
    return {
        set(uniforms){
            
            for(const key of Object.keys(uniforms)){

                const pointer = gl.getUniformLocation(shader.program,key)
                const Value=uniforms[key]
            
                if((! pointer)||(Value===undefined)){
                    return
                }
                
                
                
                if((Value!==undefined)&&(Value["constructor"]["name"]==="WebGLTexture")){
                    gl.uniform1i(pointer,sampler)
                    gl.activeTexture(gl.TEXTURE0+sampler)
                    gl.bindTexture(gl.TEXTURE_2D,Value)
                    sampler++
                }else{
                    
                    switch(Value.length) {
                        case 16:
                            gl.uniformMatrix4fv(pointer,null,Value)
                            break;
                        case 4:
                            gl.uniform4fv(pointer,Value)
                            break;
                        case 3:
                            gl.uniform3fv(pointer,Value)
                            break;
                        case 2:
                            gl.uniform2fv(pointer,Value)
                            break;
                        default:
                            gl.uniform1f(pointer,Value)
                      }
                    

                }
        
            }
            
        }
    }
}

  async function loadGLTF(url) {
    const gltf = await loadJSON(url);

    // load all the referenced files relative to the gltf file
    const baseURL = new URL(url, location.href);
    gltf.buffers = await Promise.all(gltf.buffers.map((buffer) => {
      const url = new URL(buffer.uri, baseURL.href);
      return loadBinary(url.href);
    }));


    gltf['data']=gltf['accessors'].map(accessor=>{
      const bufferView=gltf['bufferViews'][accessor['bufferView']]
      const TypedArray = glTypedArray(accessor['componentType'])
      return new TypedArray(
          gltf.buffers[bufferView['buffer']],
          bufferView['byteOffset'],//+ (accessor.byteOffset || 0),
          accessor['count']* NumSize(accessor['type']))
  })

    const defaultMaterial = {
      uniforms: {
        u_diffuse: [.5, .8, 1, 1],
      },
    };

    // setup meshes
    gltf.meshes.forEach((mesh) => {
      mesh.primitives.forEach((primitive) => {
        //var vao = gl.createVertexArray();
        //gl.bindVertexArray(vao);

        const attribs = {};
        let numElements;
        for (const [attribName, index] of Object.entries(primitive.attributes)) {
          const {accessor, buffer, stride} = getAccessorAndWebGLBuffer(gl, gltf, index);

          /*let key = 'a_'+attribName
          const attribute = gl.getAttribLocation(skinProgramInfo.program, key)
          if(attribute>-1){
            console.log(key)
            if(key=='a_JOINTS_0'){
              gl.vertexAttribIPointer(attribute, accessorTypeToNumComponents(accessor.type), 
              gl.UNSIGNED_SHORT, false,0,0)
            }else{
              gl.vertexAttribPointer(attribute, accessorTypeToNumComponents(accessor.type), 
                gl.FLOAT, false,0,0)
            }
            gl.enableVertexAttribArray(attribute)
          }*/

          numElements = accessor.count;
          attribs[`a_${attribName}`] = {
            buffer,
            type: accessor.componentType,
            numComponents: accessorTypeToNumComponents(accessor.type),
            stride,
            offset: accessor.byteOffset | 0,
          };
        }

        const bufferInfo = {
          attribs,
          numElements,
        };

        if (primitive.indices !== undefined) {
          const {accessor, buffer} = getAccessorAndWebGLBuffer(gl, gltf, primitive.indices);
          bufferInfo.numElements = accessor.count;
          bufferInfo.indices = buffer;
          bufferInfo.elementType = accessor.componentType;
        }

        primitive.bufferInfo = bufferInfo;

        // make a VAO for this primitive
        // NOTE: This is problematic. In order to automagically
        // setup the attributes we need a ProgramInfo since a ProgramInfo
        // contains the type and size of each attribute. But, for this to
        // work for all situation we'd need a ProgramInfo that uses every
        // possible attribute and for all similar attributes to use the
        // same location. For this particular situation we use
        // skinProgramInfo and above where we compiled the shaders we
        // set the locations but for a larger program we'd need some other
        // solution
        //primitive.vao = vao//twgl.createVAOFromBufferInfo(gl, skinProgramInfo, primitive.bufferInfo);
        primitive.vao = createVAOFromBufferInfo(gl, skinProgramInfo, primitive.bufferInfo);

        // save the material info for this primitive
        primitive.material = gltf.materials && gltf.materials[primitive.material] || defaultMaterial;
      });
    });

    const skinNodes = [];
    const origNodes = gltf.nodes;
    gltf.nodes = gltf.nodes.map((n) => {
      const {name, skin, mesh, translation, rotation, scale} = n;
      const trs = new TRS(translation, rotation, scale);
      const node = new Node(trs, name);
      const realMesh = gltf.meshes[mesh];
      if (skin !== undefined) {
        skinNodes.push({node, mesh: realMesh, skinNdx: skin});
      } else if (realMesh) {
        node.drawables.push(new MeshRenderer(realMesh));
      }
      return node;
    });

    // setup skins
    gltf.skins = gltf.skins.map((skin) => {
      const joints = skin.joints.map(ndx => gltf.nodes[ndx]);
      const {stride, array} = getAccessorTypedArrayAndStride(gl, gltf, skin.inverseBindMatrices);
      return new Skin(joints, array);
    });

    // Add SkinRenderers to nodes with skins
    for (const {node, mesh, skinNdx} of skinNodes) {
      node.drawables.push(new SkinRenderer(mesh, gltf.skins[skinNdx]));
    }

    // arrange nodes into graph
    gltf.nodes.forEach((node, ndx) => {
      const children = origNodes[ndx].children;
      if (children) {
        addChildren(gltf.nodes, node, children);
      }
    });

    // setup scenes
    for (const scene of gltf.scenes) {
      scene.root = new Node(new TRS(), scene.name);
      addChildren(gltf.nodes, scene.root, scene.nodes);
    }

    return gltf;
  }

  function addChildren(nodes, node, childIndices) {
    childIndices.forEach((childNdx) => {
      const child = nodes[childNdx];
      child.setParent(node);
    });
  }

  async function loadFile(url, typeFunc) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`could not load: ${url}`);
    }
    return await response[typeFunc]();
  }

  async function loadBinary(url) {
    return loadFile(url, 'arrayBuffer');
  }

  async function loadJSON(url) {
    return loadFile(url, 'json');
  }

  const gltf = await loadGLTF('https://webgl2fundamentals.org/webgl/resources/models/killer_whale/whale.CYCLES.gltf');

  function degToRad(deg) {
    return deg * Math.PI / 180;
  }




  function lerp(input, target, percent) {
    input += (target - input)*percent;
    return input
}
function lerpVec(input, target, percent) {
    return input.map((p,i)=>lerp(input[i], target[i], percent))
}






  const origMatrices = new Map();
  function animSkin(model, skin, a) {


    /*for (let i = 0; i < skin.joints.length; ++i) {
      const joint = skin.joints[i];
      // if there is no matrix saved for model joint
      if (!origMatrices.has(joint)) {
        // save a matrix for joint
        origMatrices.set(joint, joint.source.getMatrix());
      }
      // get the original matrix
      const origMatrix = origMatrices.get(joint);
      // rotate it
      const m = m4.xRotate(origMatrix, a);
      // decompose it back into position, rotation, scale
      // into the joint
      m4.decompose(m, joint.source.position, joint.source.rotation, joint.source.scale);
    }*/


    if(!model.animData){
      model.animData = {frames: 15}
  }


  if(!model['animations']||!model.animations.length){
      return
  }

  //const skin=model['skins'][0]

      const animation=model.animations[0]



      model.animData.frame = model.animData.frame??0
      model.animData.percent=model.animData.percent??0.0;
      model.animData.frames=model.animData.frames??0
      model.animData.time=model.animData.time??0

  
      const frametime=0.035

      model.animData.percent=model.animData.time/frametime
      model.animData.time+=a//delta

      const calcPercent=()=>{
      while(model.animData.percent>frametime){
          model.animData.frame++
          model.animData.time=0
          model.animData.percent=0
      }
      }
      if(model.animData.frame>model.animData.frames){
      model.animData.frame=0
      calcPercent()
      }
      if(model.animData.time>frametime){
      calcPercent()
      }
      

    for(const channel of animation['channels']){
      const target=channel['target']
      const joint=model.nodes[target['node']]
      const sampler=animation['samplers'][channel['sampler']]
      
      const fname={'translation':'position','rotation':'rotation','scale':'scale',}[channel.target['path']]
      const len=(fname==='rotation')?4:3

     
      const accessor=model.data[sampler['output']]
      
      if(!model.animData.frames){
          model.animData.frames=accessor.length/len
      }
  
      if(fname==='rotation'){
          var value=[accessor[(model.animData.frame*len)+0],accessor[(model.animData.frame*len)+1],accessor[(model.animData.frame*len)+2],accessor[(model.animData.frame*len)+3]]
      }else{
          var value=[accessor[(model.animData.frame*len)+0],accessor[(model.animData.frame*len)+1],accessor[(model.animData.frame*len)+2]]
      }

      let fromVal=[value[0],value[1],value[2],value[3],]
  
      const newVal=lerpVec(fromVal, value, model.animData.percent)

      joint.source[fname]=newVal

      //console.log('ENND JOINT',joint)
      }


      










  }

  function render(time) {
    time *= 0.001;  // convert to seconds

    //twgl.resizeCanvasToDisplaySize(gl.canvas);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.clearColor(.1, .1, .1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const fieldOfViewRadians = degToRad(60);
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const projection = mat4.create()
    mat4.perspective(projection, fieldOfViewRadians, aspect, 1, 2000);

    const cameraPosition = [10, 0, -5];
    const target = [0, 0, -10];
    // for debugging .. see article
    // const cameraPosition = [5, 0, 5];
    // const target = [0, 0, 0];
    const up = [0, 1, 0];
    // Compute the camera's matrix using look at.
    const camera = mat4.create()
    mat4.lookAt(camera, cameraPosition, target, up);

    // Make a view matrix from the camera matrix.
    const view = mat4.create()
    mat4.invert(view, camera);


    animSkin(gltf, gltf.skins[0], time);

    let outVec = vec3.create()
    const sharedUniforms = {
      u_lightDirection: vec3.normalize(outVec,[-1, 3, 5]),
    };

    function renderDrawables(node) {
      for (const drawable of node.drawables) {
        drawable.render(node, projection, view, sharedUniforms);
      }
    }

    for (const scene of gltf.scenes) {
      // updatte all world matices in the scene.
      scene.root.updateWorldMatrix();
      // walk the scene and render all renderables
      scene.root.traverse(renderDrawables);
    }

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

main();
