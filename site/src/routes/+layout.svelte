<script>
    import Menu from "$lib/menu.svelte"
    import "$lib/app.css"
</script>

<div class="page">
    <div id="sidebar" width="25%">
        <Menu />
        <p>Layout</p>
    </div>
    <div id="content" width="75%">
        <slot/>
    </div>
</div>
