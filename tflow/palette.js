/* Colors for the canvases of TransformerFlow, read from the tokens of ../tokens.css.
   Canvas has no var(), so this is the one place that turns a token into a fillStyle;
   the rest of the page reads the same tokens through its stylesheet. */
(function(root){
'use strict';
const cache=new Map();
/** Value of a custom property as the stylesheet declares it; '' until the stylesheet has loaded. */
function token(name){if(!cache.has(name)){const value=getComputedStyle(document.documentElement).getPropertyValue(name).trim();if(!value)return value;cache.set(name,value);}return cache.get(name);}
/** A '#rrggbb' token as [r, g, b]. */
function rgb(name){const hex=token(name);if(!/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(hex))throw Error('Token '+name+' is not a hex color: '+hex);const full=hex.length===4?'#'+[...hex.slice(1)].map(c=>c+c).join(''):hex;return [1,3,5].map(i=>parseInt(full.slice(i,i+2),16));}
/** A hex token with an alpha, as an rgba() string; alpha('--nn-accent', .16) is the accent at 16 %. */
function alpha(name,a){const [r,g,b]=rgb(name);return 'rgba('+r+', '+g+', '+b+', '+a+')';}
root.TFlowPalette={token,rgb,alpha};
})(window);
