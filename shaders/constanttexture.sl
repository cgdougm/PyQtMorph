/* 
 * Constant texture blend
 *
 *
 */


surface
constanttexture ( float blend = 1;
		 string texturename = ""; )
{
  color Ct;

  if (texturename != "")
       Ct = color texture (texturename);
  else Ct = 1;

  Oi = Os * blend;
  Ci = Os * blend * Ct;
}

