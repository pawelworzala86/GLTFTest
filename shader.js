//var SHADERCACHE = {}

function CreateShader(gl,vertCode,fragCode){
   //var name = 'default'

   //if(SHADERCACHE[name]){
   //   return SHADERCACHE[name]
   //}

    //var vertCode = await get('/shaders/'+name+'.vert')
     //    var fragCode = await get('/shaders/'+name+'.frag')

      function CreateShaderA(type,code){
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

         var vertShader = CreateShaderA(gl.VERTEX_SHADER,vertCode)
         var fragShader = CreateShaderA(gl.FRAGMENT_SHADER,fragCode);

         var program = gl.createProgram();
         gl.attachShader(program, vertShader);
         gl.attachShader(program, fragShader);
         gl.linkProgram(program);

   //SHADERCACHE[name] = {program}

    return {program}
    
}