/* 
 * Morph
 *
 *
 */


surface
morph ( float blend = 0;
		 string imageA = ""; 
		 string imageB = ""; 
 )
{
  color Ct;
  color CtA;
  color CtB;

  if (imageA != "")
       CtA = color texture (imageA);
  else CtA = 1;

  if (imageB != "")
       CtB = color texture (imageB);
  else CtB = 1;

  Oi = Os;
  Ci = Os * ( blend * CtB + (1-blend) * CtA);
}

